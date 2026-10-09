import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useDebounced, useFetch } from '../hooks/useFetch'
import AlbumArt from '../components/AlbumArt'
import SongRow from '../components/SongRow'
import SectionHeader from '../components/SectionHeader'
import { EmptyState, ErrorMessage, Spinner } from '../components/Status'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const debounced = useDebounced(query.trim(), 300)

  const results = useFetch(
    debounced.length >= 1 ? () => api.search(debounced, 12) : null, [debounced])

  return (
    <div className="mx-auto max-w-4xl">
      <SectionHeader title="Search" subtitle="Songs, artists, albums and genres" />
      <input
        autoFocus
        className="input mb-6 text-base"
        placeholder="What do you want to hear?"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      {results.loading && <Spinner label="Searching…" />}
      {results.error && <ErrorMessage message={results.error} />}

      {debounced && results.data && (
        <div className="space-y-8">
          {results.data.songs.length === 0 && results.data.artists.length === 0 &&
            results.data.albums.length === 0 && results.data.genres.length === 0 && (
              <EmptyState title={`No results for “${debounced}”`}
                          hint="Try a different title, artist or genre." />
          )}

          {results.data.artists.length > 0 && (
            <section>
              <h3 className="mb-2 text-sm font-semibold text-zinc-400">Artists</h3>
              <div className="flex flex-wrap gap-3">
                {results.data.artists.map((artist) => (
                  <Link key={artist.id} to={`/artists/${artist.id}`}
                        className="card flex w-40 flex-col items-center gap-2 text-center transition hover:border-accent/30">
                    <AlbumArt id={artist.id} title={artist.name} size={96} rounded="rounded-full" />
                    <span className="w-full truncate text-sm font-semibold">{artist.name}</span>
                    <span className="text-[11px] text-zinc-500">{artist.genres.slice(0, 2).join(' · ')}</span>
                  </Link>
                ))}
              </div>
            </section>
          )}

          {results.data.songs.length > 0 && (
            <section>
              <h3 className="mb-2 text-sm font-semibold text-zinc-400">Songs</h3>
              <div className="card divide-y divide-white/5 p-2">
                {results.data.songs.map((song, i) => (
                  <SongRow key={song.id} item={song} index={i} />
                ))}
              </div>
            </section>
          )}

          {results.data.albums.length > 0 && (
            <section>
              <h3 className="mb-2 text-sm font-semibold text-zinc-400">Albums</h3>
              <div className="flex flex-wrap gap-3">
                {results.data.albums.map((album) => (
                  <Link key={album.id} to={`/artists/${album.artist_id}`}
                        className="card flex w-40 flex-col gap-2 transition hover:border-accent/30">
                    <AlbumArt id={album.id} title={album.title} size={120} className="w-full" />
                    <span className="truncate text-sm font-semibold">{album.title}</span>
                    <span className="text-[11px] text-zinc-500">{album.release_year ?? '—'}</span>
                  </Link>
                ))}
              </div>
            </section>
          )}

          {results.data.genres.length > 0 && (
            <section>
              <h3 className="mb-2 text-sm font-semibold text-zinc-400">Genres</h3>
              <div className="flex flex-wrap gap-2">
                {results.data.genres.map((g) => (
                  <Link key={g.id} to="/discover" className="chip px-4 py-1.5">{g.name}</Link>
                ))}
              </div>
            </section>
          )}
        </div>
      )}

      {!debounced && (
        <EmptyState title="Start typing to search the catalog" />
      )}
    </div>
  )
}
