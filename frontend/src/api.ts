const API_BASE_URL = "http://localhost:8000";

export type DebateStatus = "pending" | "running" | "completed" | "failed";

export type PipelineStage =
  | "round_1"
  | "round_2"
  | "round_3"
  | "round_4"
  | "judging"
  | "verification"
  | "consensus"
  | "synthesis";

// Order and labels must match backend/app/api/debates.py's PIPELINE_STAGES.
export const PIPELINE_STAGES: { key: PipelineStage; label: string }[] = [
  { key: "round_1", label: "Independent Answers" },
  { key: "round_2", label: "Cross Critique" },
  { key: "round_3", label: "Counterarguments" },
  { key: "round_4", label: "Revision" },
  { key: "judging", label: "Judging" },
  { key: "verification", label: "Fact Verification" },
  { key: "consensus", label: "Consensus" },
  { key: "synthesis", label: "Final Synthesis" },
];

export interface DebateSummary {
  id: number;
  question: string;
  status: DebateStatus;
  stage: PipelineStage | null;
}

export interface CouncilVerdict {
  question: string;
  final_answer: string;
  consensus_level: "strong_consensus" | "partial_consensus" | "no_consensus";
  consensus_explanation: string;
  confidence: number;
  strongest_agent: string;
  strongest_agent_score: number;
  key_disagreements: string[];
  agent_positions: Record<string, string>;
  claims_supported: number;
  claims_checked: number;
}

export interface DebateDetail extends DebateSummary {
  verdict: CouncilVerdict | null;
  error: string | null;
}

export async function createDebate(question: string): Promise<DebateSummary> {
  const response = await fetch(`${API_BASE_URL}/debates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    throw new Error(`Failed to create debate: ${response.status}`);
  }
  return response.json();
}

export async function getDebate(id: number): Promise<DebateDetail> {
  const response = await fetch(`${API_BASE_URL}/debates/${id}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch debate: ${response.status}`);
  }
  return response.json();
}
