interface Props {
  title: string
  subtitle?: string
  action?: React.ReactNode
}

export default function SectionHeader({ title, subtitle, action }: Props) {
  return (
    <div className="mb-3 flex items-end justify-between gap-4">
      <div>
        <h2 className="text-lg font-bold tracking-tight">{title}</h2>
        {subtitle && <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}
