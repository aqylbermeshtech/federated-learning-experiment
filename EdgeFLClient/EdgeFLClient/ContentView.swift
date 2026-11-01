//
//  ContentView.swift
//  EdgeFLClient
//
//  Created by Nurtore on 01.11.2026.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var modelManager = FLModelManager()
    @State private var strokes: [[CGPoint]] = []
    @State private var currentStroke: [CGPoint] = []
    @State private var selectedLabel: Int = 3
    @State private var statusMessage: String = "Draw a digit (0–9) below"
    @State private var roundNumber: Int = 1
    @State private var serverBaseURL: String = "http://127.0.0.1:8000"

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                Text("Edge Federated Client")
                    .font(.title3)
                    .bold()

                // Server Configuration Header
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        Image(systemName: "network")
                            .foregroundColor(.blue)
                            .font(.caption)
                        TextField("http://127.0.0.1:8000", text: $serverBaseURL)
                            .font(.caption)
                            .padding(6)
                            .background(Color(.tertiarySystemBackground))
                            .cornerRadius(8)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                    }
                    HStack(spacing: 8) {
                        Button("Simulator (127.0.0.1)") {
                            serverBaseURL = "http://127.0.0.1:8000"
                        }
                        .font(.caption2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(6)

                        Button("Device (192.168.100.187)") {
                            serverBaseURL = "http://192.168.100.187:8000"
                        }
                        .font(.caption2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(6)
                    }
                }
                .padding(.horizontal, 4)

                // Status & Prediction Banner
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Round \(roundNumber) • Loss: \(String(format: "%.4f", modelManager.trainingLoss))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(statusMessage)
                            .font(.subheadline)
                            .bold()
                            .lineLimit(2)
                    }
                    Spacer()
                    if let pred = modelManager.predictedDigit {
                        VStack(alignment: .trailing) {
                            Text("Predicted: \(pred)")
                                .font(.headline)
                                .foregroundColor(.blue)
                            Text(String(format: "%.1f%% conf", modelManager.predictionConfidence))
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(12)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)

                // 280x280 Drawing Canvas (10x scale of 28x28 MNIST)
                ZStack {
                    Color.black
                    
                    Path { path in
                        for stroke in strokes {
                            guard let first = stroke.first else { continue }
                            path.move(to: first)
                            for point in stroke.dropFirst() {
                                path.addLine(to: point)
                            }
                        }
                        if let first = currentStroke.first {
                            path.move(to: first)
                            for point in currentStroke.dropFirst() {
                                path.addLine(to: point)
                            }
                        }
                    }
                    .stroke(Color.white, style: StrokeStyle(lineWidth: 22, lineCap: .round, lineJoin: .round))
                }
                .frame(width: 280, height: 280)
                .cornerRadius(16)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.gray.opacity(0.4), lineWidth: 1)
                )
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let pt = value.location
                            if pt.x >= 0, pt.x <= 280, pt.y >= 0, pt.y <= 280 {
                                currentStroke.append(pt)
                            }
                        }
                        .onEnded { _ in
                            if !currentStroke.isEmpty {
                                strokes.append(currentStroke)
                                currentStroke = []
                                modelManager.predict(strokes: strokes)
                            }
                        }
                )

                // Canvas Controls
                HStack(spacing: 8) {
                    Button(action: clearCanvas) {
                        Label("Clear", systemImage: "trash")
                            .font(.subheadline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(Color(.secondarySystemBackground))
                            .foregroundColor(.primary)
                            .cornerRadius(10)
                    }

                    Button(action: { modelManager.predict(strokes: strokes) }) {
                        Label("Predict", systemImage: "sparkles")
                            .font(.subheadline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(Color.purple.opacity(0.15))
                            .foregroundColor(.purple)
                            .cornerRadius(10)
                    }
                    .disabled(strokes.isEmpty)

                    Button(action: resetModel) {
                        Label("Reset", systemImage: "arrow.counterclockwise")
                            .font(.subheadline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(Color.red.opacity(0.12))
                            .foregroundColor(.red)
                            .cornerRadius(10)
                    }
                }

                // Ground Truth Label Picker
                VStack(alignment: .leading, spacing: 6) {
                    Text("True Digit Label (for Training): \(selectedLabel)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Picker("Digit Label", selection: $selectedLabel) {
                        ForEach(0..<10, id: \.self) { digit in
                            Text("\(digit)").tag(digit)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                // Train & Submit Button
                Button(action: trainOnDrawingAndSubmit) {
                    if modelManager.isTraining {
                        ProgressView()
                            .tint(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                    } else {
                        Label("Train on Digit '\(selectedLabel)' & Submit", systemImage: "arrow.up.circle.fill")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                    }
                }
                .background(strokes.isEmpty || modelManager.isTraining ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(12)
                .disabled(strokes.isEmpty || modelManager.isTraining)
            }
            .padding(16)
        }
    }

    private func clearCanvas() {
        strokes.removeAll()
        currentStroke.removeAll()
        modelManager.predictedDigit = nil
        statusMessage = "Canvas cleared"
    }

    private func resetModel() {
        modelManager.resetModel()
        clearCanvas()
        statusMessage = "Model reset to baseline."
    }

    private func trainOnDrawingAndSubmit() {
        guard let batch = modelManager.generateBatchFromDrawing(strokes: strokes, label: selectedLabel) else {
            statusMessage = "Failed to rasterize drawing"
            return
        }

        statusMessage = "Training on digit '\(selectedLabel)'..."
        modelManager.trainLocally(with: batch) { weights in
            // Re-run prediction immediately to observe local weight shift
            modelManager.predict(strokes: strokes)
            statusMessage = "Sending \(weights.count) weights to coordinator..."
            sendWeightsToServer(weights: weights)
        }
    }

    private func sendWeightsToServer(weights: [Float]) {
        let trimmedBase = serverBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: "\(trimmedBase)/submit_weights") else {
            statusMessage = "Invalid Server URL: \(serverBaseURL)"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let payload: [String: Any] = [
            "client_id": "iOS_Canvas_Client",
            "round": roundNumber,
            "num_samples": 32,
            "weights": weights
        ]

        guard let httpBody = try? JSONSerialization.data(withJSONObject: payload) else { return }
        request.httpBody = httpBody

        URLSession.shared.dataTask(with: request) { data, _, error in
            DispatchQueue.main.async {
                if let error = error {
                    if self.serverBaseURL.contains("127.0.0.1") {
                        self.statusMessage = "Server unreachable. Tip: If running on a physical iPhone, tap 'Device (192.168.100.187)' above. Also ensure 'python bridge_server.py' is running on your Mac."
                    } else {
                        self.statusMessage = "Server unreachable: \(error.localizedDescription). Ensure 'python bridge_server.py' is running on your Mac."
                    }
                    return
                }

                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    let status = json["status"] as? String ?? ""
                    if status == "aggregated" {
                        let newRound = json["new_round"] as? Int ?? (self.roundNumber + 1)
                        self.roundNumber = newRound
                        self.statusMessage = "Round \(newRound - 1) aggregated! Advanced to Round \(newRound)."
                    } else if status == "waiting_for_more_clients" {
                        let pool = json["current_pool"] as? Int ?? 1
                        let req = json["required_clients"] as? Int ?? 2
                        self.statusMessage = "Weights submitted! Waiting for clients (\(pool)/\(req))."
                    } else {
                        self.roundNumber += 1
                        self.statusMessage = "Round \(self.roundNumber - 1) complete."
                    }
                } else {
                    self.roundNumber += 1
                    self.statusMessage = "Round synced with coordinator."
                }
            }
        }.resume()
    }
}
