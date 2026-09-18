import { useEffect, useState } from "react";
import { createDebate, getDebate, type DebateDetail } from "./api";
import { AgentCard } from "./AgentCard";
import { ProgressChecklist } from "./ProgressChecklist";

function App() {
  const [question, setQuestion] = useState("");
  const [debate, setDebate] = useState<DebateDetail | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!debate || debate.status === "completed" || debate.status === "failed") {
      return;
    }
    const interval = setInterval(async () => {
      try {
        const updated = await getDebate(debate.id);
        setDebate(updated);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to poll debate status");
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [debate]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setSubmitting(true);
    setError(null);
    setDebate(null);
    try {
      const summary = await createDebate(question);
      setDebate({ ...summary, verdict: null, error: null });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start debate");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-3xl font-semibold text-gray-800 text-center mb-8">AI Council</h1>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask your question..."
            rows={3}
            className="border border-gray-300 rounded-lg p-3 resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <button
            type="submit"
            disabled={submitting || !question.trim()}
            className="bg-blue-600 text-white rounded-lg py-2 font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700"
          >
            {submitting ? "Starting..." : "Start Debate"}
          </button>
        </form>

        {error && <p className="text-red-600 mt-4">{error}</p>}

        {debate && (
          <div className="mt-8 bg-white rounded-lg shadow p-6">
            <p className="text-sm text-gray-500 mb-2">Debate #{debate.id}</p>
            <p className="font-medium mb-4">{debate.question}</p>

            {(debate.status === "pending" || debate.status === "running") && (
              <div>
                <p className="text-blue-600 mb-4">
                  {debate.status === "pending" ? "Starting..." : "Debate in progress..."}
                </p>
                <ProgressChecklist currentStage={debate.stage} />
              </div>
            )}

            {debate.status === "failed" && (
              <p className="text-red-600">Debate failed: {debate.error}</p>
            )}

            {debate.status === "completed" && debate.verdict && (
              <div className="flex flex-col gap-6">
                <div className="flex flex-col gap-3">
                  <p className="text-sm font-semibold uppercase text-gray-500">
                    {debate.verdict.consensus_level.replace("_", " ")}
                  </p>
                  <p className="text-lg">{debate.verdict.final_answer}</p>
                  <p className="text-sm text-gray-600">
                    Confidence: {debate.verdict.confidence.toFixed(1)}/100
                  </p>
                  <p className="text-sm text-gray-600">
                    Claims verified: {debate.verdict.claims_supported}/{debate.verdict.claims_checked}
                  </p>
                </div>

                <div className="flex flex-col gap-3">
                  <p className="text-sm font-semibold text-gray-500">Individual Answers</p>
                  {debate.verdict.agents.map((agent) => (
                    <AgentCard key={agent.name} agent={agent} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
