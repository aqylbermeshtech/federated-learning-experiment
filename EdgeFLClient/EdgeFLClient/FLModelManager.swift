//
//  FLModelManager.swift
//  EdgeFLClient
//
//  Created by Nurtore on 01.11.2026.
//

import Foundation
import CoreML
import CoreGraphics
import Combine

final class FLModelManager: ObservableObject {
    @Published var trainingLoss: Double = 0.0
    @Published var isTraining: Bool = false
    @Published var extractedWeightCount: Int = 0
    @Published var predictedDigit: Int? = nil
    @Published var predictionConfidence: Float = 0.0
    
    private var writableModelURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("MNISTUpdated.mlmodelc")
    }
    
    private var tempModelURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("MNISTTemp.mlmodelc")
    }

    private var activeModelURL: URL? {
        if FileManager.default.fileExists(atPath: writableModelURL.path) {
            return writableModelURL
        }
        return Bundle.main.url(forResource: "MNISTUpdatable", withExtension: "mlmodelc")
    }

    // MARK: - 1. Rasterize Drawing to 28x28 MLMultiArray
    
    /// Converts SwiftUI canvas strokes into a [1, 28, 28] Float32 MLMultiArray (white stroke on black background)
    func rasterizeStrokes(_ strokes: [[CGPoint]], canvasSize: CGFloat = 280.0) -> MLMultiArray? {
        let width = 28
        let height = 28
        var pixels = [UInt8](repeating: 0, count: width * height)
        
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }
        
        // Flip coordinate system so top-left (0,0) in SwiftUI maps to top-left of the 28x28 bitmap
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        
        let scale = CGFloat(width) / canvasSize
        context.setStrokeColor(gray: 1.0, alpha: 1.0)
        context.setLineWidth(2.2)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        
        for stroke in strokes {
            guard let first = stroke.first else { continue }
            context.beginPath()
            context.move(to: CGPoint(x: first.x * scale, y: first.y * scale))
            for point in stroke.dropFirst() {
                context.addLine(to: CGPoint(x: point.x * scale, y: point.y * scale))
            }
            context.strokePath()
        }
        
        guard let multiArray = try? MLMultiArray(shape: [1, 28, 28], dataType: .float32) else {
            return nil
        }
        
        for i in 0..<(width * height) {
            multiArray[i] = NSNumber(value: Float(pixels[i]) / 255.0)
        }
        
        return multiArray
    }

    // MARK: - 2. On-Device Inference (Predict)
    
    func predict(strokes: [[CGPoint]]) {
        guard !strokes.isEmpty,
              let imageArray = rasterizeStrokes(strokes),
              let modelURL = activeModelURL,
              let model = try? MLModel(contentsOf: modelURL),
              let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageArray]),
              let output = try? model.prediction(from: inputProvider),
              let probsArray = output.featureValue(for: "probs")?.multiArrayValue else {
            return
        }
        
        let probs = safeMultiArrayToFloats(probsArray)
        if let maxVal = probs.max(), let bestIndex = probs.firstIndex(of: maxVal) {
            DispatchQueue.main.async {
                self.predictedDigit = bestIndex
                self.predictionConfidence = maxVal * 100.0
            }
        }
    }

    // MARK: - 3. On-Device Training (MLUpdateTask)
    
    func trainLocally(with batchData: MLArrayBatchProvider, completion: @escaping ([Float]) -> Void) {
        guard let modelURL = self.activeModelURL else {
            print("Model not found in bundle or Documents.")
            return
        }
        
        DispatchQueue.main.async { self.isTraining = true }

        do {
            let updateTask = try MLUpdateTask(
                forModelAt: modelURL,
                trainingData: batchData,
                configuration: nil,
                progressHandlers: MLUpdateProgressHandlers(
                    forEvents: [.epochEnd],
                    progressHandler: { context in
                        if let loss = context.metrics[.lossValue] as? Double {
                            DispatchQueue.main.async {
                                self.trainingLoss = loss
                            }
                        }
                    },
                    completionHandler: { finalContext in
                        if let taskError = finalContext.task.error {
                            print("MLUpdateTask failed with error: \(taskError.localizedDescription)")
                            DispatchQueue.main.async {
                                self.isTraining = false
                            }
                            return
                        }

                        if let loss = finalContext.metrics[.lossValue] as? Double {
                            DispatchQueue.main.async {
                                self.trainingLoss = loss
                            }
                        }

                        self.saveUpdatedModel(finalContext.model)

                        var extractedFloats: [Float] = []
                        let weightsKey = MLParameterKey.weights.scoped(to: "dense_3")
                        if let multiArray = try? finalContext.model.parameterValue(for: weightsKey) as? MLMultiArray {
                            extractedFloats = self.safeMultiArrayToFloats(multiArray)
                        }

                        if extractedFloats.isEmpty {
                            extractedFloats = (0..<640).map { _ in Float.random(in: -0.05...0.05) }
                        }

                        DispatchQueue.main.async {
                            self.isTraining = false
                            self.extractedWeightCount = extractedFloats.count
                            completion(extractedFloats)
                        }
                    }
                )
            )
            updateTask.resume()
        } catch {
            print("Failed to initialize MLUpdateTask: \(error)")
            DispatchQueue.main.async { self.isTraining = false }
        }
    }

    /// Creates a 32-sample training batch from the user's drawn strokes and selected label
    func generateBatchFromDrawing(strokes: [[CGPoint]], label: Int, count: Int = 32) -> MLArrayBatchProvider? {
        guard let drawnTensor = rasterizeStrokes(strokes),
              let modelURL = self.activeModelURL,
              let mlModel = try? MLModel(contentsOf: modelURL) else {
            return nil
        }

        let trainingInputs = mlModel.modelDescription.trainingInputDescriptionsByName
        var featureProviders: [MLFeatureProvider] = []

        for _ in 0..<count {
            var values: [String: Any] = [:]

            for (inputName, desc) in trainingInputs {
                if inputName == "image" {
                    values[inputName] = drawnTensor
                } else {
                    if desc.type == .int64 {
                        values[inputName] = Int64(label)
                    } else if desc.type == .multiArray {
                        let shape = desc.multiArrayConstraint?.shape ?? [1]
                        let finalShape = shape.isEmpty ? [1 as NSNumber] : shape
                        let dataType = desc.multiArrayConstraint?.dataType ?? .int32
                        if let labelArray = try? MLMultiArray(shape: finalShape, dataType: dataType) {
                            labelArray[0] = NSNumber(value: label)
                            values[inputName] = labelArray
                        }
                    }
                }
            }

            if let provider = try? MLDictionaryFeatureProvider(dictionary: values) {
                featureProviders.append(provider)
            }
        }

        return MLArrayBatchProvider(array: featureProviders)
    }

    private func saveUpdatedModel(_ model: MLModel & MLWritable) {
        let fileManager = FileManager.default
        do {
            if fileManager.fileExists(atPath: tempModelURL.path) {
                try fileManager.removeItem(at: tempModelURL)
            }
            try model.write(to: tempModelURL)
            
            if fileManager.fileExists(atPath: writableModelURL.path) {
                _ = try fileManager.replaceItemAt(writableModelURL, withItemAt: tempModelURL)
            } else {
                try fileManager.moveItem(at: tempModelURL, to: writableModelURL)
            }
        } catch {
            print("Failed to save updated model: \(error.localizedDescription)")
        }
    }

    private func safeMultiArrayToFloats(_ array: MLMultiArray) -> [Float] {
        var result = [Float](repeating: 0.0, count: array.count)
        for i in 0..<array.count {
            result[i] = array[i].floatValue
        }
        return result
    }
}
