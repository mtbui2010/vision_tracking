'use client';

import { useEffect, useRef } from 'react';
import type { PredictionFrame } from '@/services/api';

interface Props {
  videoUrl: string;
  predictions: PredictionFrame[];
  width: number;
  height: number;
}

function colorFor(id: number): string {
  const hue = (id * 137.508) % 360;
  return `hsl(${hue}, 70%, 60%)`;
}

export function TrackOverlay({ videoUrl, predictions, width, height }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    const tick = () => {
      const fps = 30;
      const idx = Math.min(predictions.length - 1, Math.floor(video.currentTime * fps));
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const frame = predictions[idx];
      if (frame) {
        for (let i = 0; i < frame.ids.length; i++) {
          const [x1, y1, x2, y2] = frame.bboxes[i];
          const id = frame.ids[i];
          ctx.strokeStyle = colorFor(id);
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillStyle = colorFor(id);
          ctx.fillRect(x1, y1 - 16, 36, 16);
          ctx.fillStyle = '#000';
          ctx.font = '12px monospace';
          ctx.fillText(`#${id}`, x1 + 4, y1 - 4);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [predictions]);

  return (
    <div className="relative" style={{ width, height }}>
      <video ref={videoRef} src={videoUrl} controls width={width} height={height} className="block" />
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="absolute inset-0 pointer-events-none"
      />
    </div>
  );
}
