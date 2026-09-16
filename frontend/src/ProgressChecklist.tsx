import { PIPELINE_STAGES, type PipelineStage } from "./api";

interface Props {
  currentStage: PipelineStage | null;
}

export function ProgressChecklist({ currentStage }: Props) {
  const currentIndex = currentStage
    ? PIPELINE_STAGES.findIndex((s) => s.key === currentStage)
    : -1;

  return (
    <ul className="flex flex-col gap-2">
      {PIPELINE_STAGES.map((stage, index) => {
        const isDone = currentIndex > index;
        const isCurrent = currentIndex === index;

        return (
          <li key={stage.key} className="flex items-center gap-2 text-sm">
            <span
              className={
                isDone
                  ? "text-green-600"
                  : isCurrent
                    ? "text-blue-600"
                    : "text-gray-300"
              }
            >
              {isDone ? "✓" : isCurrent ? "●" : "○"}
            </span>
            <span
              className={
                isCurrent ? "text-gray-900 font-medium" : isDone ? "text-gray-600" : "text-gray-400"
              }
            >
              {stage.label}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
