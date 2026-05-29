const BASE = process.env.NEXT_PUBLIC_API_BASE || '/api';

export type TrackerName = 'sort' | 'deepsort' | 'bytetrack';
export type ComputeTarget = 'local' | 'cloud' | 'hybrid';

export type JobStatus = 'queued' | 'running' | 'done' | 'failed';

export interface Job {
  id: string;
  kind: string;
  status: JobStatus;
  result?: TrackingResult;
  error?: string;
}

export interface TrackingResult {
  tracker: TrackerName;
  frames: number;
  fps: number;
  predictions: PredictionFrame[];
}

export interface PredictionFrame {
  frame: number;
  ids: number[];
  bboxes: [number, number, number, number][];
}

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface AssociationResponse {
  iou_matrix: number[][];
  cost_matrix: number[][];
  matches: [number, number][];
  unmatched_detections: number[];
  unmatched_tracks: number[];
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) throw new Error(`${path} -> ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  trackers: () => jsonFetch<{ trackers: TrackerName[] }>('/tracking/trackers'),

  submitJob: async (video: File, tracker: TrackerName, computeTarget?: ComputeTarget) => {
    const form = new FormData();
    form.append('video', video);
    form.append('tracker', tracker);
    if (computeTarget) form.append('compute_target', computeTarget);
    return jsonFetch<Job>('/tracking/jobs', { method: 'POST', body: form });
  },

  getJob: (id: string) => jsonFetch<Job>(`/tracking/jobs/${id}`),

  submitCompare: async (video: File, trackers: TrackerName[]) => {
    const form = new FormData();
    form.append('video', video);
    form.append('trackers', trackers.join(','));
    return jsonFetch<{ id: string; status: string }>('/compare/jobs', { method: 'POST', body: form });
  },

  kalmanStep: (bbox: BBox, steps: number = 1) =>
    jsonFetch<{ predicted_bbox: number[]; state: number[]; covariance_diag: number[] }>(
      '/algorithm/kalman',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox, steps }),
      },
    ),

  associate: (detections: BBox[], tracks: BBox[], iouThreshold: number = 0.3) =>
    jsonFetch<AssociationResponse>('/algorithm/associate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detections, tracks, iou_threshold: iouThreshold }),
    }),

  streamJob: (id: string, onMessage: (msg: string) => void) => {
    const proto = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = typeof window !== 'undefined' ? window.location.host : 'localhost:3000';
    const ws = new WebSocket(`${proto}://${host}${BASE}/tracking/jobs/${id}/stream`);
    ws.onmessage = (e) => onMessage(e.data);
    return () => ws.close();
  },
};
