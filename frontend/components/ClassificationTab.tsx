"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type Category = "bug" | "feature" | "question" | "documentation";
type ModelName =
  | "LogisticRegression"
  | "DecisionTree"
  | "RandomForest"
  | "GradientBoosting"
  | "MLP"
  | "LSTM"
  | "GRU";

type SimilarIssue = {
  title: string;
  repo: string;
  similarity: number;
  url: string;
};

type ModelMetric = {
  model: ModelName;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
};

type AnalyzeResponse = {
  category: Category;
  confidence: number;
  similar_issues: SimilarIssue[];
  model_metrics: ModelMetric[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const examplePrompts = [
  "Cannot read properties of undefined",
  "Feature request: support new frontend framework version",
  "How can I prevent unnecessary re-renders?",
  "Documentation is missing state management examples",
];

const modelLabels: Record<ModelName, string> = {
  LogisticRegression: "로지스틱 회귀",
  DecisionTree: "결정트리",
  RandomForest: "랜덤 포레스트",
  GradientBoosting: "그래디언트 부스팅",
  MLP: "다층 퍼셉트론",
  LSTM: "LSTM",
  GRU: "GRU",
};

const categoryLabels: Record<Category, string> = {
  bug: "Bug",
  feature: "Feature",
  question: "Question",
  documentation: "Documentation",
};

export default function ClassificationTab() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");

  async function analyzeIssue() {
    setStatus("loading");
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });
      if (!response.ok) throw new Error("Analyze request failed");
      setResult((await response.json()) as AnalyzeResponse);
      setStatus("idle");
    } catch {
      setResult(null);
      setStatus("error");
    }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
      <Card>
        <CardHeader>
          <CardTitle>Issue 입력</CardTitle>
          <CardDescription>
            프론트엔드 오류 메시지, GitHub Issue 내용, 개발 질문을 입력하세요.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <textarea
            className="min-h-56 w-full resize-none rounded-md border border-zinc-300 bg-zinc-50 p-4 text-sm leading-6 outline-none ring-emerald-500 transition focus:ring-2"
            id="issue-input"
            onChange={(event) => setInput(event.target.value)}
            placeholder="오류 메시지나 질문을 입력하세요."
            value={input}
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {examplePrompts.map((prompt) => (
              <button
                className="rounded-md border border-zinc-200 bg-white px-3 py-2 text-xs font-semibold text-zinc-600 transition hover:border-emerald-600 hover:text-emerald-700"
                key={prompt}
                onClick={() => setInput(prompt)}
                type="button"
              >
                {prompt}
              </button>
            ))}
          </div>
          <button
            className="mt-4 h-11 w-full rounded-md bg-zinc-950 px-5 text-sm font-bold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-zinc-400 sm:w-auto"
            disabled={status === "loading" || input.trim().length === 0}
            onClick={analyzeIssue}
            type="button"
          >
            {status === "loading" ? "분석 중..." : "Issue 분석 실행"}
          </button>
          {status === "error" && (
            <p className="mt-3 text-sm text-red-600">
              API 서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인하세요.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>분석 결과</CardTitle>
          <CardDescription>
            로지스틱 회귀 분류 결과와 TF-IDF 코사인 유사도 기반 추천 Issue를 보여줍니다.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {status === "loading" ? (
            <div className="rounded-md border border-zinc-200 bg-zinc-50 p-5">
              <p className="text-sm font-semibold text-zinc-700">입력 내용을 분석하고 있습니다.</p>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-zinc-200">
                <div className="h-full w-1/2 rounded-full bg-emerald-500" />
              </div>
            </div>
          ) : result === null ? (
            <div className="rounded-md border border-zinc-200 bg-zinc-50 p-5">
              <p className="text-sm text-zinc-500">분석을 실행하면 결과가 표시됩니다.</p>
            </div>
          ) : (
            <div className="space-y-5">
              <section className="rounded-md border border-zinc-200 bg-zinc-50 p-5">
                <p className="text-xs font-semibold uppercase text-zinc-500">분류 카테고리</p>
                <div className="mt-3 flex items-end justify-between gap-4">
                  <p className="text-3xl font-bold text-zinc-950">{categoryLabels[result.category]}</p>
                  <p className="text-sm font-semibold text-emerald-700">
                    신뢰도 {Math.round(result.confidence * 100)}%
                  </p>
                </div>
                <div className="mt-3 flex flex-wrap gap-2 border-t border-zinc-200 pt-3">
                  <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700">
                    분류 · 로지스틱 회귀
                  </span>
                  <span className="inline-flex items-center gap-1 rounded-full bg-zinc-100 px-2.5 py-0.5 text-xs font-medium text-zinc-600">
                    추천 · TF-IDF 코사인 유사도
                  </span>
                </div>
              </section>

              <section>
                <h3 className="text-base font-bold text-zinc-950">유사 Issue TOP 3</h3>
                <div className="mt-3 space-y-3">
                  {result.similar_issues.map((issue, index) => (
                    <a
                      className="block rounded-md border border-zinc-200 bg-white p-4 transition hover:border-emerald-600 hover:bg-zinc-50"
                      href={issue.url}
                      key={`${issue.repo}-${index}`}
                      rel="noreferrer"
                      target="_blank"
                    >
                      <p className="font-bold leading-6 text-zinc-950">
                        {index + 1}. {issue.title}
                      </p>
                      <p className="mt-1 text-sm text-zinc-500">{issue.repo}</p>
                      <p className="mt-2 break-all text-xs font-semibold text-emerald-700">
                        유사도 {Math.round(issue.similarity * 100)}% / {issue.url}
                      </p>
                    </a>
                  ))}
                </div>
              </section>
            </div>
          )}
        </CardContent>
      </Card>

      {result !== null && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>모델 성능 비교</CardTitle>
            <CardDescription>
              결정트리 계열 모델과 신경망 모델의 분류 성능을 비교합니다.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              {result.model_metrics.map((metric) => (
                <MetricBar key={metric.model} metric={metric} />
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function MetricBar({ metric }: { metric: ModelMetric }) {
  return (
    <div className="rounded-md border border-zinc-200 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="font-bold">{modelLabels[metric.model]}</span>
        <span className="text-sm text-zinc-500">F1 {Math.round(metric.f1 * 100)}%</span>
      </div>
      <div className="mt-4 h-2 rounded-full bg-zinc-100">
        <div className="h-2 rounded-full bg-emerald-500" style={{ width: `${Math.round(metric.accuracy * 100)}%` }} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-zinc-600">
        <span>정확도 {Math.round(metric.accuracy * 100)}%</span>
        <span>정밀도 {Math.round(metric.precision * 100)}%</span>
        <span>재현율 {Math.round(metric.recall * 100)}%</span>
      </div>
    </div>
  );
}
