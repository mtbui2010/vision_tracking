'use client';

import { useState } from 'react';
import { api, type Job, type TrackerName } from '@/services/api';
import { TrackOverlay } from '@/components/TrackOverlay';

const TRACKERS: TrackerName[] = ['sort', 'deepsort', 'bytetrack'];

export default function TrackerLabPage() {
  const [tracker, setTracker] = useState<TrackerName>('sort');
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [running, setRunning] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setRunning(true);
    setLogs([]);
    try {
      const created = await api.submitJob(file, tracker);
      setJob(created);
      api.streamJob(created.id, (msg) => setLogs((prev) => [...prev, msg]));
      const poll = setInterval(async () => {
        const j = await api.getJob(created.id);
        setJob(j);
        if (j.status === 'done' || j.status === 'failed') {
          clearInterval(poll);
          setRunning(false);
        }
      }, 1500);
    } catch (err) {
      setLogs([`error: ${(err as Error).message}`]);
      setRunning(false);
    }
  }

  const videoUrl = file ? URL.createObjectURL(file) : '';
  const predictions = job?.result?.predictions ?? [];

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-6">Tracker Lab</h1>

      <form onSubmit={onSubmit} className="flex flex-col gap-3 max-w-md mb-8">
        <label className="text-sm text-neutral-400">
          Tracker
          <select
            value={tracker}
            onChange={(e) => setTracker(e.target.value as TrackerName)}
            className="mt-1 w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1"
          >
            {TRACKERS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>

        <label className="text-sm text-neutral-400">
          Video
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="mt-1 w-full"
          />
        </label>

        <button
          type="submit"
          disabled={!file || running}
          className="bg-white text-black rounded px-4 py-2 disabled:opacity-40"
        >
          {running ? 'running…' : 'Run'}
        </button>
      </form>

      {videoUrl && predictions.length > 0 && (
        <TrackOverlay videoUrl={videoUrl} predictions={predictions} width={960} height={540} />
      )}

      {logs.length > 0 && (
        <pre className="mt-6 text-xs text-neutral-500 bg-neutral-950 border border-neutral-800 p-3 rounded max-h-64 overflow-auto">
          {logs.join('\n')}
        </pre>
      )}
    </div>
  );
}
