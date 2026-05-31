"use client";

import { useState, useEffect, useCallback } from "react";

interface ModelResult {
  accuracy: number;
  cv_mean: number;
  cv_std: number;
}

interface TrainResponse {
  status: string;
  message: string;
  best_model: string;
  best_accuracy: number;
  model_results: Record<string, ModelResult>;
  dataset_info: {
    total_samples: number;
    features: number;
    classes: number;
    class_distribution: Record<string, number>;
  };
}

interface PredictionResponse {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  symptoms: Record<string, number>;
  timestamp: string;
}

interface OverviewResponse {
  dataset: {
    total_samples: number;
    features: number;
    classes: number;
    class_distribution: Record<string, number>;
    sample_data: Array<Record<string, unknown>>;
  };
  model_comparison: Array<{
    model: string;
    test_accuracy: string;
    cv_mean: string;
    cv_std: string;
  }>;
  best_model: string;
}

interface EvaluationResponse {
  models: Record<string, {
    classification_report: Record<string, unknown>;
    confusion_matrix: number[][];
    accuracy: number;
  }>;
  classes: string[];
}

const API_BASE = "http://localhost:8000";

export default function Home() {
  const [activeTab, setActiveTab] = useState("overview");
  const [trained, setTrained] = useState(false);
  const [loading, setLoading] = useState(false);
  const [trainResponse, setTrainResponse] = useState<TrainResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);

  const [nSamples, setNSamples] = useState(500);
  const [testSize, setTestSize] = useState(0.2);
  const [seed, setSeed] = useState(42);

  const [symptoms, setSymptoms] = useState({
    Oropharyngeal_Burns: false,
    Teeth_Discoloration: 0,
    Abdominal_Distension: false,
    Skin_Lesions: 0,
    Melena: false,
    Hematemesis: false,
    throat_pain: false,
    dysphagia: false,
    Chest_Pain: false,
    Acidosis: false,
  });

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setTrained(data.trained);
      if (data.trained) {
        fetchOverview();
      }
    } catch {
      console.log("API not available yet");
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  const handleTrain = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          n_samples: nSamples,
          test_size: testSize,
          random_state: seed,
        }),
      });
      const data: TrainResponse = await res.json();
      setTrainResponse(data);
      setTrained(true);
      fetchOverview();
    } catch (error) {
      console.error("Training failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          Oropharyngeal_Burns: symptoms.Oropharyngeal_Burns ? 1 : 0,
          Teeth_Discoloration: symptoms.Teeth_Discoloration,
          Abdominal_Distension: symptoms.Abdominal_Distension ? 1 : 0,
          Skin_Lesions: symptoms.Skin_Lesions,
          Melena: symptoms.Melena ? 1 : 0,
          Hematemesis: symptoms.Hematemesis ? 1 : 0,
          throat_pain: symptoms.throat_pain ? 1 : 0,
          dysphagia: symptoms.dysphagia ? 1 : 0,
          Chest_Pain: symptoms.Chest_Pain ? 1 : 0,
          Acidosis: symptoms.Acidosis ? 1 : 0,
        }),
      });
      const data: PredictionResponse = await res.json();
      setPrediction(data);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchOverview = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/overview`);
      const data: OverviewResponse = await res.json();
      setOverview(data);
    } catch (error) {
      console.error("Failed to fetch overview:", error);
    }
  };

  const fetchEvaluation = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/evaluation`);
      const data: EvaluationResponse = await res.json();
      setEvaluation(data);
    } catch (error) {
      console.error("Failed to fetch evaluation:", error);
    }
  };

  useEffect(() => {
    if (activeTab === "evaluation" && trained) {
      fetchEvaluation();
    }
  }, [activeTab, trained]);

  const getProbabilityColor = (acid: string) => {
    if (acid.includes("Sulfuric")) return "#ff4d4d";
    if (acid.includes("Nitric")) return "#ffa500";
    return "#4d94ff";
  };

  return (
    <div className="flex min-h-screen">
      <aside className="w-72 p-6 bg-black/30 backdrop-blur-xl border-r border-white/10 flex flex-col">
        <h2 className="mb-6 text-xl font-bold text-pink-400 flex items-center gap-2">
          <span>⚙️</span> Dataset & Training
        </h2>

        <div className="mb-4">
          <label className="block mb-2 text-sm text-gray-300">
            Samples to generate: <span className="text-white font-medium">{nSamples}</span>
          </label>
          <input
            type="range"
            min="100"
            max="2000"
            value={nSamples}
            onChange={(e) => setNSamples(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div className="mb-4">
          <label className="block mb-2 text-sm text-gray-300">
            Test set ratio: <span className="text-white font-medium">{testSize.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.1"
            max="0.4"
            step="0.01"
            value={testSize}
            onChange={(e) => setTestSize(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div className="mb-6">
          <label className="block mb-2 text-sm text-gray-300">Random seed</label>
          <input
            type="number"
            min="1"
            max="999"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <button
          onClick={handleTrain}
          disabled={loading}
          className="w-full py-3 px-4 rounded-lg font-semibold text-white primary-gradient primary-glow transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Processing..." : "🔄 Generate & Train Models"}
        </button>

        {trainResponse && (
          <div className="mt-6 p-4 glass-card">
            <p className="text-sm text-green-400 mb-2">✓ {trainResponse.message}</p>
            <div className="text-xs text-gray-400">
              <p>Best: {trainResponse.best_model}</p>
              <p>Accuracy: {(trainResponse.best_accuracy * 100).toFixed(2)}%</p>
            </div>
          </div>
        )}

        <div className="mt-auto pt-6 border-t border-white/10">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className={`w-2 h-2 rounded-full ${trained ? "bg-green-400" : "bg-gray-500"}`}></span>
            {trained ? "Model Ready" : "Not Trained"}
          </div>
        </div>
      </aside>

      <main className="flex-1 p-8 overflow-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-pink-400 to-red-500">
            Corrosive Poison Diagnostic System
          </h1>
          <p className="text-gray-400 mt-2">AI-Powered Toxicology Analysis for Acid Identification</p>
        </header>

        <nav className="mb-6 flex gap-2">
          {["overview", "diagnostic", "evaluation"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-5 py-2.5 rounded-lg font-medium transition-all ${
                activeTab === tab
                  ? "bg-pink-600 text-white"
                  : "bg-white/5 text-gray-300 hover:bg-white/10"
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>

        {activeTab === "overview" && (
          <div className="space-y-6">
            <div className="grid grid-cols-3 gap-6">
              <div className="glass-card p-6">
                <p className="text-gray-400 text-sm mb-1">Total Samples</p>
                <p className="text-3xl font-bold text-white">
                  {overview?.dataset?.total_samples.toLocaleString() || "—"}
                </p>
              </div>
              <div className="glass-card p-6">
                <p className="text-gray-400 text-sm mb-1">Features</p>
                <p className="text-3xl font-bold text-white">
                  {overview?.dataset?.features || "—"}
                </p>
              </div>
              <div className="glass-card p-6">
                <p className="text-gray-400 text-sm mb-1">Classes</p>
                <p className="text-3xl font-bold text-white">
                  {overview?.dataset?.classes || "—"}
                </p>
              </div>
            </div>

            {overview?.model_comparison && (
              <div className="glass-card p-6">
                <h3 className="text-xl font-semibold mb-4 text-pink-300">Model Performance</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-400 border-b border-white/10">
                        <th className="pb-3 font-medium">Model</th>
                        <th className="pb-3 font-medium">Test Accuracy</th>
                        <th className="pb-3 font-medium">CV Mean</th>
                        <th className="pb-3 font-medium">CV Std</th>
                      </tr>
                    </thead>
                    <tbody>
                      {overview.model_comparison.map((row) => (
                        <tr key={row.model} className="border-b border-white/5">
                          <td className="py-3 text-white">{row.model}</td>
                          <td className="py-3 text-green-400">{row.test_accuracy}</td>
                          <td className="py-3 text-white">{row.cv_mean}</td>
                          <td className="py-3 text-gray-400">{row.cv_std}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {overview?.dataset?.class_distribution && (
              <div className="glass-card p-6">
                <h3 className="text-xl font-semibold mb-4 text-pink-300">Class Distribution</h3>
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(overview.dataset.class_distribution).map(([acid, count]) => (
                    <div key={acid} className="bg-white/5 rounded-lg p-4">
                      <p className="text-lg font-medium text-white truncate" title={acid}>
                        {acid}
                      </p>
                      <p className="text-2xl font-bold text-pink-400">{count as number}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "diagnostic" && (
          <div className="grid grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <h3 className="text-xl font-semibold mb-4 text-pink-300">Clinical Symptoms</h3>
              
              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={symptoms.Oropharyngeal_Burns}
                    onChange={(e) => setSymptoms({ ...symptoms, Oropharyngeal_Burns: e.target.checked })}
                    className="w-5 h-5 rounded accent-pink-500"
                  />
                  <span>Oropharyngeal Burns</span>
                </label>

                <div>
                  <label className="block mb-1 text-sm text-gray-400">Teeth Discoloration</label>
                  <select
                    value={symptoms.Teeth_Discoloration}
                    onChange={(e) => setSymptoms({ ...symptoms, Teeth_Discoloration: Number(e.target.value) })}
                    className="w-full"
                  >
                    <option value={0}>None</option>
                    <option value={1}>Yellow</option>
                    <option value={2}>Chalky White</option>
                  </select>
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={symptoms.Abdominal_Distension}
                    onChange={(e) => setSymptoms({ ...symptoms, Abdominal_Distension: e.target.checked })}
                    className="w-5 h-5 rounded accent-pink-500"
                  />
                  <span>Abdominal Distension</span>
                </label>

                <div>
                  <label className="block mb-1 text-sm text-gray-400">Skin Lesions</label>
                  <select
                    value={symptoms.Skin_Lesions}
                    onChange={(e) => setSymptoms({ ...symptoms, Skin_Lesions: Number(e.target.value) })}
                    className="w-full"
                  >
                    <option value={0}>None</option>
                    <option value={1}>Mild Erythema</option>
                    <option value={2}>Severe Burns</option>
                  </select>
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={symptoms.Melena}
                    onChange={(e) => setSymptoms({ ...symptoms, Melena: e.target.checked })}
                    className="w-5 h-5 rounded accent-pink-500"
                  />
                  <span>Melena (Black Stool)</span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={symptoms.Hematemesis}
                    onChange={(e) => setSymptoms({ ...symptoms, Hematemesis: e.target.checked })}
                    className="w-5 h-5 rounded accent-pink-500"
                  />
                  <span>Hematemesis (Vomiting Blood)</span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={symptoms.throat_pain}
                    onChange={(e) => setSymptoms({ ...symptoms, throat_pain: e.target.checked })}
                    className="w-5 h-5 rounded accent-pink-500"
                  />
                  <span>Throat Pain</span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={symptoms.dysphagia}
                    onChange={(e) => setSymptoms({ ...symptoms, dysphagia: e.target.checked })}
                    className="w-5 h-5 rounded accent-pink-500"
                  />
                  <span>Dysphagia (Difficulty Swallowing)</span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={symptoms.Chest_Pain}
                    onChange={(e) => setSymptoms({ ...symptoms, Chest_Pain: e.target.checked })}
                    className="w-5 h-5 rounded accent-pink-500"
                  />
                  <span>Chest Pain</span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={symptoms.Acidosis}
                    onChange={(e) => setSymptoms({ ...symptoms, Acidosis: e.target.checked })}
                    className="w-5 h-5 rounded accent-pink-500"
                  />
                  <span>Metabolic Acidosis</span>
                </label>
              </div>

              <button
                onClick={handlePredict}
                disabled={!trained || loading}
                className="w-full mt-6 py-3 px-4 rounded-lg font-semibold text-white primary-gradient primary-glow transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Analyzing..." : "🔮 Predict Corrosive Agent"}
              </button>
            </div>

            <div className="space-y-6">
              {prediction ? (
                <>
                  <div
                    className="glass-card p-8 text-center"
                    style={{
                      background: `linear-gradient(135deg, #dc3545 0%, #ff8c00 100%)`,
                    }}
                  >
                    <h3 className="text-2xl font-bold text-white mb-2">{prediction.prediction}</h3>
                    <p className="text-xl text-white/90">
                      Confidence: {prediction.confidence.toFixed(1)}%
                    </p>
                    <p className="text-sm text-white/70 mt-2">{prediction.timestamp}</p>
                  </div>

                  <div className="glass-card p-6">
                    <h3 className="text-lg font-semibold mb-4 text-pink-300">Probability by Acid Type</h3>
                    <div className="space-y-3">
                      {Object.entries(prediction.probabilities).map(([acid, prob]) => (
                        <div key={acid}>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-gray-300">{acid}</span>
                            <span className="text-white">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${prob * 100}%`,
                                backgroundColor: getProbabilityColor(acid),
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="glass-card p-8 flex items-center justify-center h-64">
                  <p className="text-gray-400">
                    {!trained
                      ? "Please train a model first"
                      : "Enter symptoms and click Predict to see results"}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "evaluation" && evaluation && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="text-xl font-semibold mb-4 text-pink-300">Classification Reports</h3>
              {Object.entries(evaluation.models).map(([modelName, data]) => (
                <div key={modelName} className="mb-6">
                  <h4 className="text-lg font-medium text-white mb-3">{modelName}</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-gray-400 border-b border-white/10">
                          <th className="pb-2 font-medium">Class</th>
                          <th className="pb-2 font-medium">Precision</th>
                          <th className="pb-2 font-medium">Recall</th>
                          <th className="pb-2 font-medium">F1-Score</th>
                          <th className="pb-2 font-medium">Support</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evaluation.classes.map((cls) => {
                          const report = data.classification_report[cls] as {
                            precision: number;
                            recall: number;
                            "f1-score": number;
                            support: number;
                          } | undefined;
                          if (!report) return null;
                          return (
                            <tr key={cls} className="border-b border-white/5">
                              <td className="py-2 text-white">{cls}</td>
                              <td className="py-2 text-green-400">{(report.precision * 100).toFixed(1)}%</td>
                              <td className="py-2 text-blue-400">{(report.recall * 100).toFixed(1)}%</td>
                              <td className="py-2 text-purple-400">{(report["f1-score"] * 100).toFixed(1)}%</td>
                              <td className="py-2 text-gray-400">{report.support}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>

            <div className="glass-card p-6">
              <h3 className="text-xl font-semibold mb-4 text-pink-300">Confusion Matrices</h3>
              <div className="grid grid-cols-2 gap-6">
                {Object.entries(evaluation.models).slice(0, 2).map(([modelName, data]) => (
                  <div key={modelName}>
                    <h4 className="text-lg font-medium text-white mb-3">{modelName}</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-center text-gray-400">
                            <th className="pb-2"></th>
                            {evaluation.classes.map((cls) => (
                              <th key={cls} className="pb-2 font-medium">{cls.split(" ")[0]}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {data.confusion_matrix.map((row, i) => (
                            <tr key={i}>
                              <td className="py-1 pr-2 text-gray-400 text-right">
                                {evaluation.classes[i].split(" ")[0]}
                              </td>
                              {row.map((cell, j) => (
                                <td
                                  key={j}
                                  className="py-1 px-2 text-center text-white"
                                >
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}