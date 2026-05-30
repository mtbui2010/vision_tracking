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

async function loadJson(name: string): Promise<BenchmarkData | null> {
  try {
    const file = path.join(process.cwd(), 'public', name);
    return JSON.parse(await fs.readFile(file, 'utf-8')) as BenchmarkData;
  } catch {
    return null;
  }
}

const BENCHMARKS = [
  {
    file: 'benchmark_yolo.json',
    title: 'MOT17 val · YOLOv8n fine-tuned on MOT17 (30 epochs, img=640)',
    subtitle: 'Detector recall jumps from FRCNN baseline; tracker IDSW for IoU-only methods explodes 3–10×; custom holds.',
  },
  {
    file: 'benchmark_dancetrack.json',
    title: 'DanceTrack val · same YOLOv8n (3 sequences: 0035, 0047, 0081)',
    subtitle: 'Same-appearance, large-motion. All methods drop sharply; ordering preserved — custom still wins on IDF1 + IDSW.',
  },
  {
    file: 'benchmark.json',
    title: 'MOT17 val · provided FRCNN detections (baseline)',
    subtitle: 'Sparse detector, FN-dominated. Apples-to-apples comparison against published numbers.',
  },
];

function fmt(n: number, digits = 4): string {
  return n.toFixed(digits);
}

function Leaderboard({ title, data, subtitle }: { title: string; data: BenchmarkData; subtitle: string }) {
  const ranked = [...data.rows].sort((a, b) => b.hota - a.hota);
  return (
    <section className="mb-10">
      <h2 className="text-lg font-medium">{title}</h2>
      <p className="text-xs text-neutral-500 mb-3">{subtitle}</p>
      <table className="text-sm border border-neutral-800 w-full max-w-3xl">
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
    </section>
  );
}

export default async function StressTestPage() {
  const loaded = await Promise.all(
    BENCHMARKS.map(async (b) => ({ ...b, data: await loadJson(b.file) })),
  );
  const present = loaded.filter((b): b is typeof b & { data: BenchmarkData } => b.data !== null);

  if (present.length === 0) {
    return (
      <div>
        <h1 className="text-2xl font-semibold mb-2">Stress test</h1>
        <p className="text-sm text-neutral-400">
          No benchmark data yet. Run
          <code className="mx-1 px-1 bg-neutral-900 rounded">make benchmark</code>
          to generate <code>frontend/public/benchmark.json</code>.
        </p>
      </div>
    );
  }

  const primary = present[0];
  return (
    <div>
      <h1 className="text-2xl font-semibold mb-2">Benchmark leaderboards</h1>
      <p className="text-sm text-neutral-400 mb-6">
        Sequence-length-weighted MOTA, mean IDF1 / HOTA. Computed by
        <code className="mx-1 px-1 bg-neutral-900 rounded">scripts/benchmark.py</code>;
        cross-checked against <code className="px-1 bg-neutral-900 rounded">py-motmetrics</code>.
      </p>

      {present.map((b) => (
        <Leaderboard key={b.file} title={b.title} subtitle={b.subtitle} data={b.data} />
      ))}

      <section>
        <h2 className="text-lg font-medium mb-2">Per-sequence breakdown ({primary.title.split(' · ')[0]})</h2>
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
              {primary.data.per_sequence.map((r, i) => (
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
      </section>

      <p className="mt-6 text-xs text-neutral-500">
        Source files: {present.map((b) => (
          <code key={b.file} className="mx-1 px-1 bg-neutral-900 rounded">{b.file}</code>
        ))}
      </p>
    </div>
  );
}
