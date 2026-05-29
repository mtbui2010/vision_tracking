import fs from 'node:fs/promises';
import path from 'node:path';

interface RowSummary {
  tracker: string;
  mota: number;
  idf1: number;
  hota: number;
  deta: number;
  assa: number;
  fp: number;
  fn: number;
  idsw: number;
  fps: number;
}

interface BenchmarkData {
  sequences: string[];
  score_threshold: number;
  rows: RowSummary[];
  per_sequence: {
    seq: string;
    tracker: string;
    mota: number;
    idf1: number;
    hota: number;
    idsw: number;
    fps: number;
  }[];
}

async function loadBenchmark(): Promise<BenchmarkData | null> {
  try {
    const file = path.join(process.cwd(), 'public', 'benchmark.json');
    const raw = await fs.readFile(file, 'utf-8');
    return JSON.parse(raw) as BenchmarkData;
  } catch {
    return null;
  }
}

function fmt(n: number, digits = 4): string {
  return n.toFixed(digits);
}

export default async function StressTestPage() {
  const data = await loadBenchmark();

  if (!data) {
    return (
      <div>
        <h1 className="text-2xl font-semibold mb-2">Stress test</h1>
        <p className="text-sm text-neutral-400">
          No benchmark data yet. Run
          <code className="mx-1 px-1 bg-neutral-900 rounded">python scripts/benchmark.py</code>
          to generate <code>frontend/public/benchmark.json</code>.
        </p>
      </div>
    );
  }

  const ranked = [...data.rows].sort((a, b) => b.hota - a.hota);

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-2">Benchmark — MOT17 val</h1>
      <p className="text-sm text-neutral-400 mb-6">
        {data.sequences.length} sequences · score threshold {data.score_threshold} ·
        aggregated by sequence-length-weighted MOTA, mean IDF1 / HOTA.
        Computed by <code className="px-1 bg-neutral-900 rounded">scripts/benchmark.py</code>;
        cross-checked against <code className="px-1 bg-neutral-900 rounded">py-motmetrics</code>.
      </p>

      <h2 className="text-lg font-medium mb-2">Leaderboard</h2>
      <table className="text-sm border border-neutral-800 w-full max-w-3xl mb-10">
        <thead className="bg-neutral-900 text-neutral-300">
          <tr>
            <th className="px-3 py-2 text-left">Tracker</th>
            <th className="px-3 py-2 text-right">HOTA ↓</th>
            <th className="px-3 py-2 text-right">DetA</th>
            <th className="px-3 py-2 text-right">AssA</th>
            <th className="px-3 py-2 text-right">MOTA</th>
            <th className="px-3 py-2 text-right">IDF1</th>
            <th className="px-3 py-2 text-right">IDSW</th>
            <th className="px-3 py-2 text-right">FPS</th>
          </tr>
        </thead>
        <tbody className="text-neutral-200">
          {ranked.map((r) => (
            <tr key={r.tracker} className="border-t border-neutral-800">
              <td className="px-3 py-2 font-medium">{r.tracker}</td>
              <td className="px-3 py-2 text-right">{fmt(r.hota)}</td>
              <td className="px-3 py-2 text-right">{fmt(r.deta)}</td>
              <td className="px-3 py-2 text-right">{fmt(r.assa)}</td>
              <td className="px-3 py-2 text-right">{fmt(r.mota)}</td>
              <td className="px-3 py-2 text-right">{fmt(r.idf1)}</td>
              <td className="px-3 py-2 text-right">{r.idsw}</td>
              <td className="px-3 py-2 text-right">{r.fps.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2 className="text-lg font-medium mb-2">Per-sequence breakdown</h2>
      <div className="overflow-auto">
        <table className="text-xs border border-neutral-800 min-w-full">
          <thead className="bg-neutral-900 text-neutral-300">
            <tr>
              <th className="px-2 py-1 text-left">Sequence</th>
              <th className="px-2 py-1 text-left">Tracker</th>
              <th className="px-2 py-1 text-right">MOTA</th>
              <th className="px-2 py-1 text-right">IDF1</th>
              <th className="px-2 py-1 text-right">HOTA</th>
              <th className="px-2 py-1 text-right">IDSW</th>
              <th className="px-2 py-1 text-right">FPS</th>
            </tr>
          </thead>
          <tbody className="text-neutral-300">
            {data.per_sequence.map((r, i) => (
              <tr key={i} className="border-t border-neutral-900">
                <td className="px-2 py-1">{r.seq}</td>
                <td className="px-2 py-1">{r.tracker}</td>
                <td className="px-2 py-1 text-right">{fmt(r.mota)}</td>
                <td className="px-2 py-1 text-right">{fmt(r.idf1)}</td>
                <td className="px-2 py-1 text-right">{fmt(r.hota)}</td>
                <td className="px-2 py-1 text-right">{r.idsw}</td>
                <td className="px-2 py-1 text-right">{r.fps.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-6 text-xs text-neutral-500">
        Source: <code className="px-1 bg-neutral-900 rounded">frontend/public/benchmark.json</code>.
      </p>
    </div>
  );
}
