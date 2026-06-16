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
          <div className="max-w-4xl">
            <p className="text-sm font-semibold text-emerald-700">Frontend Issue Finder AI</p>
            <h1 className="mt-3 text-3xl font-bold leading-tight sm:text-5xl">
              프론트엔드 Issue 분류 및 유사 사례 추천 시스템
            </h1>
            <p className="mt-4 max-w-3xl text-base leading-7 text-zinc-600">
              결정트리, 랜덤 포레스트, 그래디언트 부스팅, 신경망, KMeans를 활용해 프론트엔드 관련
              Issue를 분류하고 비슷한 사례를 찾아줍니다.
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
