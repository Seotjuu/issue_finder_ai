"use client";

import AnalyticsTab from "@/components/AnalyticsTab";
import ClassificationTab from "@/components/ClassificationTab";
import ClusteringTab from "@/components/ClusteringTab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function IssueAnalyzer() {
  return (
    <main className="min-h-screen bg-[#f6f7f4] text-zinc-950">
      <section className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-5 px-5 py-8 sm:px-8 lg:px-10">
          <div className="max-w-full">
            <p className="text-sm font-semibold text-emerald-700">
              Frontend Issue Finder AI
            </p>
            <h1 className="mt-3 text-3xl font-bold leading-tight sm:text-5xl">
              프론트엔드 Issue 분류 및 유사 사례 추천 시스템
            </h1>

            <div className="mt-5 flex flex-wrap gap-2">
              {[
                { name: "로지스틱 회귀", f1: "87%" },
                { name: "랜덤 포레스트", f1: "85%" },
                { name: "그래디언트 부스팅", f1: "85%" },
                { name: "Complement NB", f1: "82%" },
                { name: "다층 퍼셉트론", f1: "83%" },
                { name: "결정트리", f1: "80%" },
                { name: "LSTM", f1: "81%" },
                { name: "GRU", f1: "80%" },
              ].map((m) => (
                <span
                  key={m.name}
                  className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-3 py-1 text-xs font-medium text-zinc-700"
                >
                  {m.name}
                  <span className="font-bold text-emerald-600">F1 {m.f1}</span>
                </span>
              ))}
            </div>

            <p className="mt-4 text-sm leading-7 text-zinc-500">
              8가지 머신러닝·딥러닝 알고리즘을 동일 데이터셋으로 학습해 성능을 비교했습니다.
              실제 분류는 <span className="font-semibold text-zinc-700">로지스틱 회귀</span>로 수행하며,
              유사 Issue는 <span className="font-semibold text-zinc-700">TF-IDF 코사인 유사도</span>로 추천합니다.
            </p>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-6 sm:px-8 lg:px-10">
        <Tabs defaultValue="analysis">
          <TabsList className="w-full justify-start overflow-x-auto sm:w-auto">
            <TabsTrigger value="analysis">Issue 분류</TabsTrigger>
            <TabsTrigger value="clustering">군집 분석</TabsTrigger>
            <TabsTrigger value="analytics">데이터 분석</TabsTrigger>
          </TabsList>
          <TabsContent value="analysis">
            <ClassificationTab />
          </TabsContent>
          <TabsContent value="clustering">
            <ClusteringTab />
          </TabsContent>
          <TabsContent value="analytics">
            <AnalyticsTab />
          </TabsContent>
        </Tabs>
      </section>
    </main>
  );
}
