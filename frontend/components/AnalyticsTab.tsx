"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import edaData from "@/lib/eda-data.json";

type NamedCount = {
  name: string;
  count: number;
};

type BucketCount = {
  bucket: string;
  count: number;
};

const colors = ["#059669", "#2563eb", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"];

export default function AnalyticsTab() {
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <ChartCard
        data={edaData.repositoryDistribution as NamedCount[]}
        dataKey="name"
        description="수집된 Issue가 어느 저장소에 집중되어 있는지 확인합니다."
        title="Repository 분포"
      />
      <ChartCard
        data={edaData.labelDistribution as NamedCount[]}
        dataKey="name"
        description="분류 학습에 사용한 주요 Label 비율입니다."
        title="Label 분포"
      />
      <ChartCard
        data={edaData.titleLengthDistribution as BucketCount[]}
        dataKey="bucket"
        description="Issue 제목 길이를 구간별로 나눈 분포입니다."
        title="제목 길이 분포"
      />
      <ChartCard
        data={edaData.bodyLengthDistribution as BucketCount[]}
        dataKey="bucket"
        description="Issue 본문 길이를 구간별로 나눈 분포입니다."
        title="Body 길이 분포"
      />
    </div>
  );
}

function ChartCard<T extends { count: number }>({
  data,
  dataKey,
  description,
  title,
}: {
  data: T[];
  dataKey: string;
  description: string;
  title: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer height="100%" width="100%">
            <BarChart data={data} margin={{ bottom: 18, left: 0, right: 14, top: 8 }}>
              <CartesianGrid stroke="#e4e4e7" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey={dataKey} interval={0} tick={{ fontSize: 11 }} tickLine={false} />
              <YAxis tick={{ fontSize: 12 }} tickLine={false} width={46} />
              <Tooltip cursor={{ fill: "#f4f4f5" }} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell fill={colors[index % colors.length]} key={`${index}-${entry.count}`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
