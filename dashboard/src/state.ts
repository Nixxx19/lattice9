import { createContext, useContext } from "react";

export interface StreamToken {
  index: number;
  token_text: string;
  decode_worker: string;
}

export interface InferResult {
  request_id: string;
  prompt: string;
  result: string;
  tokens_generated: number;
  total_latency_ms: number;
  worker_trace: {
    worker_id: string;
    phase: string;
    layers: number[];
    latency_ms: number;
  }[];
}

export interface InferenceSession {
  prompt: string;
  maxTokens: number;
  stream: boolean;
  loading: boolean;
  error: string | null;
  result: InferResult | null;
  streamedTokens: StreamToken[];
  streamWorkers: string[];
  reshards: { token_index: number; dropped_worker: string; remaining?: string[] }[];
}

export const emptySession: InferenceSession = {
  prompt: "",
  maxTokens: 200,
  stream: true,
  loading: false,
  error: null,
  result: null,
  streamedTokens: [],
  streamWorkers: [],
  reshards: [],
};

export const InferenceCtx = createContext<{
  session: InferenceSession;
  update: (patch: Partial<InferenceSession>) => void;
}>({
  session: emptySession,
  update: () => {},
});

export function useInference() {
  return useContext(InferenceCtx);
}
