/** Small data-fetching hooks. */
import { useCallback, useEffect, useRef, useState } from 'react'

interface FetchState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

/** Declarative fetch hook with dependency-based refetching. */
export function useFetch<T>(fetcher: (() => Promise<T>) | null, deps: unknown[] = []) {
  const [state, setState] = useState<FetchState<T>>({ data: null, loading: !!fetcher, error: null })
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const reload = useCallback(() => {
    const fn = fetcherRef.current
    if (!fn) return
    setState((s) => ({ ...s, loading: true, error: null }))
    fn()
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((error: Error) => setState({ data: null, loading: false, error: error.message }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { ...state, reload }
}

/** Debounced value (search input). */
export function useDebounced<T>(value: T, delayMs = 300): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(timer)
  }, [value, delayMs])
  return debounced
}
