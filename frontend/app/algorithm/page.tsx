'use client';

import { useState } from 'react';
import { api, type BBox } from '@/services/api';

export default function AlgorithmPage() {
  const [detections, setDetections] = useState<BBox[]>([
    { x1: 100, y1: 100, x2: 200, y2: 300 },
    { x1: 300, y1: 100, x2: 400, y2: 300 },
  ]);
  const [tracks, setTracks] = useState<BBox[]>([
    { x1: 110, y1: 110, x2: 210, y2: 310 },
    { x1: 310, y1: 110, x2: 410, y2: 310 },
  ]);
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.associate>> | null>(null);

  async function run() {
    setResult(await api.associate(detections, tracks, 0.3));
  }

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-2">Algorithm inspector</h1>
      <p className="text-sm text-neutral-400 mb-6">
        Walk through one frame of association. Inputs are pixel-coordinate bboxes. Output shows the IoU
        matrix the Hungarian algorithm operates on, and the resulting matches.
      </p>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <BBoxList title="Detections" boxes={detections} onChange={setDetections} />
        <BBoxList title="Tracks (predicted)" boxes={tracks} onChange={setTracks} />
      </div>

      <button onClick={run} className="bg-white text-black rounded px-4 py-2">Associate</button>

      {result && (
        <div className="mt-8 space-y-4">
          <Matrix title="IoU matrix (det × trk)" data={result.iou_matrix} />
          <div>
            <h3 className="text-sm font-medium text-neutral-300 mb-1">Matches</h3>
            <ul className="text-sm text-neutral-400">
              {result.matches.map(([d, t], i) => (
                <li key={i}>det {d} ↔ track {t}</li>
              ))}
              {!result.matches.length && <li>(none)</li>}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function BBoxList({
  title,
  boxes,
  onChange,
}: {
  title: string;
  boxes: BBox[];
  onChange: (b: BBox[]) => void;
}) {
  return (
    <div>
      <div className="text-sm font-medium mb-2">{title}</div>
      {boxes.map((b, i) => (
        <div key={i} className="flex gap-1 mb-1 text-xs">
          {(['x1', 'y1', 'x2', 'y2'] as const).map((k) => (
            <input
              key={k}
              type="number"
              value={b[k]}
              onChange={(e) => {
                const v = parseFloat(e.target.value || '0');
                const next = [...boxes];
                next[i] = { ...next[i], [k]: v };
                onChange(next);
              }}
              className="w-16 bg-neutral-900 border border-neutral-800 rounded px-1 py-0.5"
            />
          ))}
        </div>
      ))}
    </div>
  );
}

function Matrix({ title, data }: { title: string; data: number[][] }) {
  return (
    <div>
      <div className="text-sm font-medium mb-1">{title}</div>
      <table className="text-xs border border-neutral-800">
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td
                  key={j}
                  className="border border-neutral-800 px-2 py-1"
                  style={{ background: `rgba(96, 165, 250, ${v})` }}
                >
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
