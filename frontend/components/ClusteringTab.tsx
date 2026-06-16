"use client";

import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type Category = "bug" | "feature" | "question" | "documentation";

type ClusterPoint = {
  id: number;
  x: number;
  y: number;
  clusterId: number;
  category: Category;
  title: string;
  repo: string;
};

type RepresentativeIssue = {
  title: string;
  repo: string;
  url: string;
};

type Cluster = {
  clusterId: number;
  name: string;
  dominantCategory: Category;
  count: number;
  color: string;
  labelDistribution: Partial<Record<Category, number>>;
  keywords: string[];
  representativeIssues: RepresentativeIssue[];
};

type ClusterData = {
  summary: {
    totalIssues: number;
    sourceIssues?: number;
    samplePerLabel?: number;
    clusterCount: number;
    silhouetteScore: number;
    algorithm?: string;
  };
  points: ClusterPoint[];
  clusters: Cluster[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const CATEGORY_LABELS: Record<Category, string> = {
  bug: "Bug",
  feature: "Feature",
  question: "Question",
  documentation: "Documentation",
};

export default function ClusteringTab() {
  const [data, setData] = useState<ClusterData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/cluster-data`)
      .then((response) => response.json())
      .then((clusterData: ClusterData) => {
        const sortedClusters = [...clusterData.clusters].sort((a, b) => a.clusterId - b.clusterId);
        setData({ ...clusterData, clusters: sortedClusters });
        setSelectedId(sortedClusters[0]?.clusterId ?? null);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const pointsByCluster = useMemo(() => {
    if (!data) return new Map<number, ClusterPoint[]>();
    const grouped = new Map<number, ClusterPoint[]>();
    for (const cluster of data.clusters) grouped.set(cluster.clusterId, []);
    for (const point of data.points) grouped.get(point.clusterId)?.push(point);
    return grouped;
  }, [data]);

  const selectedCluster = data?.clusters.find((cluster) => cluster.clusterId === selectedId) ?? data?.clusters[0];

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <p className="text-sm text-zinc-500">군집 데이터를 불러오는 중...</p>
      </div>
    );
  }

  if (!data || !selectedCluster) {
    return (
      <div className="flex h-64 items-center justify-center">
        <p className="text-sm text-red-500">군집 데이터를 불러오지 못했습니다. 백엔드 서버를 확인하세요.</p>
      </div>
    );
  }

  return (
    <div className="grid gap-5">
      <div className="grid gap-4 md:grid-cols-4">
        <SummaryCard label="샘플 Issue" value={data.summary.totalIssues.toLocaleString()} />
        <SummaryCard label="라벨별 샘플" value={String(data.summary.samplePerLabel ?? 100)} />
        <SummaryCard label="군집 수" value={String(data.summary.clusterCount)} />
        <SummaryCard label="실루엣 점수" value={data.summary.silhouetteScore.toFixed(3)} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Card>
          <CardHeader>
            <CardTitle>KMeans 군집 지도</CardTitle>
            <CardDescription>
              라벨별 100개 Issue를 샘플링한 뒤 TF-IDF, TruncatedSVD, KMeans로 4개 군집을 생성합니다.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[430px] w-full">
              <ResponsiveContainer height="100%" width="100%">
                <ScatterChart margin={{ bottom: 14, left: 0, right: 18, top: 8 }}>
                  <CartesianGrid stroke="#e4e4e7" strokeDasharray="3 3" />
                  <XAxis dataKey="x" name="SVD 1" tick={{ fontSize: 11 }} type="number" />
                  <YAxis dataKey="y" name="SVD 2" tick={{ fontSize: 11 }} type="number" />
                  <ZAxis range={[7, 7]} />
                  <Tooltip content={<PointTooltip clusters={data.clusters} />} cursor={{ strokeDasharray: "3 3" }} />
                  {data.clusters.map((cluster) => (
                    <Scatter
                      data={pointsByCluster.get(cluster.clusterId) ?? []}
                      fill={cluster.color}
                      fillOpacity={selectedId === cluster.clusterId ? 0.78 : 0.3}
                      isAnimationActive={false}
                      key={cluster.clusterId}
                      name={cluster.name}
                      onClick={() => setSelectedId(cluster.clusterId)}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {data.clusters.map((cluster) => (
                <button
                  className={`rounded-md border px-3 py-2 text-sm font-semibold transition ${
                    selectedId === cluster.clusterId
                      ? "border-zinc-950 bg-zinc-950 text-white"
                      : "border-zinc-200 bg-zinc-50 text-zinc-700 hover:border-zinc-400"
                  }`}
                  key={cluster.clusterId}
                  onClick={() => setSelectedId(cluster.clusterId)}
                  type="button"
                >
                  <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: cluster.color }} />
                  {cluster.name}
                  <span className="ml-2 text-xs opacity-70">{cluster.count}</span>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        <ClusterDetail cluster={selectedCluster} />
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {data.clusters.map((cluster) => (
          <button
            className={`rounded-md border bg-white p-4 text-left transition ${
              selectedId === cluster.clusterId ? "border-zinc-950 shadow-sm" : "border-zinc-200 hover:border-zinc-400"
            }`}
            key={cluster.clusterId}
            onClick={() => setSelectedId(cluster.clusterId)}
            type="button"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="font-bold">{cluster.name}</span>
              <span className="h-3 w-3 rounded-full" style={{ backgroundColor: cluster.color }} />
            </div>
            <p className="mt-2 text-sm text-zinc-600">
              대표 라벨: <span className="font-semibold text-zinc-950">{CATEGORY_LABELS[cluster.dominantCategory]}</span>
            </p>
            <p className="mt-1 text-sm text-zinc-500">{cluster.count} issues</p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {cluster.keywords.slice(0, 5).map((keyword) => (
                <span className="rounded bg-zinc-100 px-2 py-1 text-xs font-semibold text-zinc-600" key={keyword}>
                  {keyword}
                </span>
              ))}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function ClusterDetail({ cluster }: { cluster: Cluster }) {
  const total = Math.max(cluster.count, 1);

  return (
    <Card>
      <CardHeader>
        <CardTitle>{cluster.name}</CardTitle>
        <CardDescription>이 KMeans 군집 중심에 가장 가까운 Issue를 보여줍니다.</CardDescription>
      </CardHeader>
      <CardContent>
        <div>
          <p className="text-xs font-semibold uppercase text-zinc-500">라벨 구성</p>
          <div className="mt-3 space-y-2">
            {(Object.keys(CATEGORY_LABELS) as Category[]).map((category) => {
              const count = cluster.labelDistribution[category] ?? 0;
              const percent = Math.round((count / total) * 100);
              return (
                <div key={category}>
                  <div className="flex justify-between text-xs text-zinc-600">
                    <span>{CATEGORY_LABELS[category]}</span>
                    <span>{percent}%</span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-zinc-100">
                    <div className="h-2 rounded-full bg-zinc-900" style={{ width: `${percent}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="mt-5">
          <p className="text-xs font-semibold uppercase text-zinc-500">주요 키워드</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {cluster.keywords.slice(0, 10).map((keyword) => (
              <span className="rounded-md bg-zinc-100 px-2.5 py-1 text-xs font-semibold text-zinc-700" key={keyword}>
                {keyword}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-5 space-y-3">
          <p className="text-xs font-semibold uppercase text-zinc-500">대표 Issue</p>
          {cluster.representativeIssues.slice(0, 4).map((issue) => (
            <a
              className="block rounded-md border border-zinc-200 bg-zinc-50 p-3 transition hover:border-emerald-600 hover:bg-white"
              href={issue.url}
              key={`${issue.repo}-${issue.title}`}
              rel="noreferrer"
              target="_blank"
            >
              <p className="text-sm font-semibold leading-6 text-zinc-950">{issue.title}</p>
              <p className="mt-1 text-xs text-zinc-500">{issue.repo}</p>
            </a>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="pt-5">
        <p className="text-xs font-semibold text-zinc-500">{label}</p>
        <p className="mt-2 break-words text-2xl font-bold leading-tight text-zinc-950">{value}</p>
      </CardContent>
    </Card>
  );
}

function PointTooltip({
  active,
  clusters,
  payload,
}: {
  active?: boolean;
  clusters: Cluster[];
  payload?: Array<{ payload: ClusterPoint }>;
}) {
  if (!active || !payload?.[0]) return null;
  const point = payload[0].payload;
  const cluster = clusters.find((item) => item.clusterId === point.clusterId);

  return (
    <div className="max-w-72 rounded-md border border-zinc-200 bg-white p-3 text-sm shadow-sm">
      <p className="font-bold" style={{ color: cluster?.color ?? "#111827" }}>
        {cluster?.name ?? `Cluster ${point.clusterId + 1}`} / {CATEGORY_LABELS[point.category]}
      </p>
      <p className="mt-1 text-zinc-700">{point.title}</p>
      <p className="mt-1 text-xs text-zinc-500">{point.repo}</p>
    </div>
  );
}
