import Link from 'next/link';

const CARDS = [
  {
    href: '/tracker-lab',
    title: 'Tracker Lab',
    body: 'Upload a video, pick a tracker, watch tracks render in real time.',
  },
  {
    href: '/compare',
    title: 'Compare',
    body: 'Run SORT / DeepSORT / ByteTrack side-by-side. Live MOTA / IDF1 / FPS.',
  },
  {
    href: '/algorithm',
    title: 'Algorithm',
    body: 'Inspect Kalman predict + Hungarian cost matrix one frame at a time.',
  },
  {
    href: '/stress-test',
    title: 'Stress Test',
    body: 'ID-switch leaderboard on heavy-occlusion clips.',
  },
];

export default function Home() {
  return (
    <div>
      <section className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight">Tracker Lab</h1>
        <p className="mt-3 text-neutral-400 max-w-2xl">
          Multi-object tracking research playground. SORT, DeepSORT, ByteTrack and a custom variant —
          all implemented from scratch and benchmarked on MOT17 / MOT20 / DanceTrack.
        </p>
      </section>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {CARDS.map((c) => (
          <Link
            key={c.href}
            href={c.href}
            className="block rounded-lg border border-neutral-800 hover:border-neutral-600 p-5"
          >
            <div className="font-medium">{c.title}</div>
            <div className="mt-2 text-sm text-neutral-400">{c.body}</div>
          </Link>
        ))}
      </div>
    </div>
  );
}
