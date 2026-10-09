import { useState } from 'react'
import { useApp } from '../store/AppContext'

const GENRE_OPTIONS = [
  'electronic', 'rock', 'pop', 'hip hop', 'jazz', 'classical', 'metal',
  'r&b', 'folk', 'indie', 'ambient', 'latin',
]

/** First-run onboarding: name + genre preferences (cold-start signals). */
export default function OnboardingModal() {
  const { loginAsDemo, finishOnboarding } = useApp()
  const [name, setName] = useState('')
  const [genres, setGenres] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const toggle = (genre: string) =>
    setGenres((prev) => (prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]))

  const submit = async () => {
    if (!name.trim()) {
      setError('Please enter a display name')
      return
    }
    setBusy(true)
    setError(null)
    try {
      await finishOnboarding(name.trim(), genres)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not create user — is the backend running?')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="card w-full max-w-md bg-surface-850 p-6 shadow-2xl">
        <h1 className="text-xl font-bold">Welcome to TuneFlow 🎧</h1>
        <p className="mt-1 text-sm text-zinc-400">
          Pick a few genres and we will cold-start your recommendations instantly.
        </p>

        <label className="mt-5 block text-xs font-medium text-zinc-400">Display name</label>
        <input
          className="input mt-1"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Alex"
          maxLength={40}
        />

        <p className="mt-4 text-xs font-medium text-zinc-400">Genres you like (optional)</p>
        <div className="mt-2 flex flex-wrap gap-2">
          {GENRE_OPTIONS.map((genre) => (
            <button
              key={genre}
              onClick={() => toggle(genre)}
              className={`chip cursor-pointer transition ${genres.includes(genre) ? 'chip-active' : ''}`}
            >
              {genre}
            </button>
          ))}
        </div>

        {error && <p className="mt-3 text-xs text-rose-400">{error}</p>}

        <div className="mt-6 flex items-center justify-between gap-3">
          <button className="btn btn-ghost text-xs" disabled={busy} onClick={() => void loginAsDemo()}>
            Explore as demo user
          </button>
          <button className="btn btn-primary" disabled={busy} onClick={() => void submit()}>
            {busy ? 'Creating…' : 'Start listening'}
          </button>
        </div>
      </div>
    </div>
  )
}
