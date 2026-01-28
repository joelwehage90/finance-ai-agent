create table if not exists public.report_runs (
  id uuid primary key default gen_random_uuid(),
  customer_id text null,
  period_year int not null,
  period_month int not null,
  status text not null default 'draft',
  created_at timestamptz not null default now()
);
