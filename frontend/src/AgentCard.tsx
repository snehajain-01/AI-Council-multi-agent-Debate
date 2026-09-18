import { useState } from "react";
import type { AgentSummary } from "./api";

interface Props {
  agent: AgentSummary;
}

export function AgentCard({ agent }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex justify-between items-baseline mb-2">
        <p className="font-semibold text-gray-800">{agent.name}</p>
        <p className="text-xs text-gray-400">{agent.model}</p>
      </div>

      <p className="text-gray-800 mb-3">{agent.position}</p>

      <div className="flex gap-4 text-sm text-gray-600 mb-3">
        <span>Confidence: {(agent.confidence * 100).toFixed(0)}%</span>
        <span>Judge score: {agent.judge_score.toFixed(1)}/100</span>
      </div>

      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="text-sm text-blue-600 hover:underline"
      >
        {expanded ? "Hide reasoning" : "Show reasoning"}
      </button>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-100 flex flex-col gap-3 text-sm text-gray-700">
          <div>
            <p className="font-medium text-gray-500 mb-1">Reasoning</p>
            <p>{agent.reasoning}</p>
          </div>

          {agent.key_arguments.length > 0 && (
            <div>
              <p className="font-medium text-gray-500 mb-1">Key arguments</p>
              <ul className="list-disc list-inside">
                {agent.key_arguments.map((arg, i) => (
                  <li key={i}>{arg}</li>
                ))}
              </ul>
            </div>
          )}

          {agent.weaknesses.length > 0 && (
            <div>
              <p className="font-medium text-gray-500 mb-1">Acknowledged weaknesses</p>
              <ul className="list-disc list-inside">
                {agent.weaknesses.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          {agent.changes_from_original.length > 0 && (
            <div>
              <p className="font-medium text-gray-500 mb-1">Changed after debate</p>
              <ul className="list-disc list-inside">
                {agent.changes_from_original.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
