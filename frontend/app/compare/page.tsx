'use client';

import { useState } from 'react';
import { api, type TrackerName } from '@/services/api';

const ALL: TrackerName[] = ['sort', 'deepsort', 'bytetrack'];

export default function ComparePage() {
  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<TrackerName[]>(ALL);
  const [jobId, setJobId] = useState<string | null>(null);

  function toggle(name: TrackerName) {
    setSelected((sel) => (sel.includes(name) ? sel.filter((n) => n !== name) : [...sel, name]));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || selected.length === 0) return;
    const { id } = await api.submitCompare(file, selected);
    setJobId(id);
  }

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-6">Compare trackers</h1>

      <form onSubmit={onSubmit} className="flex flex-col gap-4 max-w-md mb-8">
        <fieldset className="flex gap-3">
          {ALL.map((t) => (
            <label key={t} className="flex items-center gap-1 text-sm">
              <input type="checkbox" checked={selected.includes(t)} onChange={() => toggle(t)} />
              {t}
            </label>
          ))}
        </fieldset>

        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />

        <button
          type="submit"
          disabled={!file || selected.length === 0}
          className="bg-white text-black rounded px-4 py-2 disabled:opacity-40"
        >
          Run comparison
        </button>
      </form>

      {jobId && (
        <p className="text-sm text-neutral-400">
          Comparison job <code>{jobId}</code> queued. Live grid + metrics rendering will land in week 8.
        </p>
      )}
    </div>
  );
}
