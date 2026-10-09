import { Link, useParams } from 'react-router-dom'
import { api } from '../services/api'
import { useFetch } from '../hooks/useFetch'
import AlbumArt from '../components/AlbumArt'
import SongRow from '../components/SongRow'
import SectionHeader from '../components/SectionHeader'
import { ErrorMessage, Spinner } from '../components/Status'

export default function ArtistPage() {
  const { artistId } = useParams<{ artistId: string }>()
  const artist = useFetch(artistId ? () => api.getArtist(artistId) : null, [artistId])

  if (artist.loading && !artist.data) return <Spinner label="Loading artist…" />
  if (artist.error || !artist.data) return <ErrorMessage message={artist.error ?? 'Artist not found'} />

  const a = artist.data

  return (
    <div className="mx-auto max-w-5xl">
      <header className="mb-8 flex flex-col items-start gap-5 sm:flex-row sm:items-end">
        <AlbumArt id={a.id} title={a.name} imageUrl={a.image_url} size={160} rounded="rounded-full"
                  className="shadow-2xl" />
        <div>
          <p className="text-xs uppercase tracking-widest text-zinc-500">Artist</p>
          <h1 className="mt-1 text-4xl font-extrabold tracking-tight">{a.name}</h1>
          <p className="mt-2 text-sm text-zinc-400">
            {Math.round(a.followers).toLocaleString()} followers
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {a.genres.map((g) => <span key={g} className="chip">{g}</span>)}
          </div>
        </div>
      </header>

      <section className="mb-10">
        <SectionHeader title="Popular Tracks" subtitle="Ranked by TuneFlow listening activity" />
        <div className="card divide-y divide-white/5 p-2">
          {a.top_songs.map((song, i) => <SongRow key={song.id} item={song} index={i} />)}
          {a.top_songs.length === 0 && <p className="p-4 text-sm text-zinc-500">No tracks yet.</p>}
        </div>
      </section>

      {a.albums.length > 0 && (
        <section className="mb-10">
          <SectionHeader title="Albums" />
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            {a.albums.map((album) => (
              <div key={album.id} className="card flex flex-col gap-2">
                <AlbumArt id={album.id} title={album.title} imageUrl={album.image_url} size={140}
                          className="w-full" />
                <span className="truncate text-xs font-semibold">{album.title}</span>
                <span className="text-[11px] text-zinc-500">{album.release_year ?? '—'}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <p className="text-xs text-zinc-600">
        <Link to="/search" className="hover:text-accent-soft">Search</Link> for more from the catalog.
      </p>
    </div>
  )
}
