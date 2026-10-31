-- FraudGuard — Supabase schema. Run once in Supabase → SQL Editor → New query → Run.
-- Every table is private per user through row-level security (RLS).

create table if not exists public.scorings (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  mode text not null check (mode in ('creditcard', 'upi')),
  input jsonb not null,
  score double precision not null,
  risk double precision not null,
  decision text not null check (decision in ('APPROVE', 'STEP-UP OTP', 'BLOCK')),
  policy jsonb,
  label smallint,
  created_at timestamptz not null default now()
);

create table if not exists public.batch_runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  mode text not null check (mode in ('creditcard', 'upi')),
  file_name text not null,
  n_rows integer not null,
  summary jsonb not null,
  storage_path text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.policies (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  mode text not null check (mode in ('creditcard', 'upi')),
  name text not null,
  costs jsonb not null,
  t_otp double precision not null,
  t_block double precision not null,
  created_at timestamptz not null default now()
);

create table if not exists public.audit_log (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  action text not null,
  detail jsonb,
  created_at timestamptz not null default now()
);

-- Row-level security: a user sees and changes only their own rows.
alter table public.scorings   enable row level security;
alter table public.batch_runs enable row level security;
alter table public.policies   enable row level security;
alter table public.audit_log  enable row level security;

create policy "own scorings"   on public.scorings   for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own batch runs" on public.batch_runs for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own policies"   on public.policies   for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
-- the audit log is append-only for users: they can read and insert, never edit or delete
create policy "read own audit"   on public.audit_log for select using (auth.uid() = user_id);
create policy "insert own audit" on public.audit_log for insert with check (auth.uid() = user_id);

-- Server-side limits (the app also checks): 50 scorings and 5 batch runs per user.
create or replace function public.enforce_limit() returns trigger language plpgsql security definer set search_path = public as $$
declare n integer; lim integer := tg_argv[0]::integer;
begin
  execute format('select count(*) from public.%I where user_id = $1', tg_table_name) into n using new.user_id;
  if n >= lim then raise exception 'Limit reached: % rows in %', lim, tg_table_name; end if;
  return new;
end $$;
drop trigger if exists limit_scorings on public.scorings;
create trigger limit_scorings before insert on public.scorings for each row execute function public.enforce_limit(50);
drop trigger if exists limit_batches on public.batch_runs;
create trigger limit_batches before insert on public.batch_runs for each row execute function public.enforce_limit(5);

create index if not exists scorings_user_idx   on public.scorings (user_id, created_at desc);
create index if not exists batch_runs_user_idx on public.batch_runs (user_id, created_at desc);
create index if not exists audit_user_idx      on public.audit_log (user_id, created_at desc);

-- Private storage bucket for scored batch CSVs: files live under <user id>/...
insert into storage.buckets (id, name, public) values ('batches', 'batches', false) on conflict (id) do nothing;
create policy "own batch files read"   on storage.objects for select using (bucket_id = 'batches' and auth.uid()::text = (storage.foldername(name))[1]);
create policy "own batch files write"  on storage.objects for insert with check (bucket_id = 'batches' and auth.uid()::text = (storage.foldername(name))[1]);
create policy "own batch files delete" on storage.objects for delete using (bucket_id = 'batches' and auth.uid()::text = (storage.foldername(name))[1]);
