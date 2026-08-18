/** Polling loop for background tasks, extracted from useData so the retry
 * semantics are testable without a store or a real fetch. */

export interface TaskStatus {
  task_id: string
  status: 'running' | 'completed' | 'cancelled' | 'error'
  result?: Record<string, unknown>
  error?: string
  progress?: number // 0..1 fraction, if the task reports it
  message?: string // human-readable progress message
}

export interface PollOptions {
  intervalMs?: number
  /** Consecutive non-404 failures tolerated before giving up. */
  maxTransientFailures?: number
  onProgress?: (status: TaskStatus) => void
  /** Injectable for tests. */
  sleep?: (ms: number) => Promise<void>
}

/** Shown when a poll 404s: the server answered, but no longer knows the task.
 * The task registry lives in the backend process, so this means that process
 * was replaced mid-run — in dev, uvicorn --reload does that on any .py edit
 * (including edits made by a Claude session or a file-sync touch). */
export const TASK_LOST_MESSAGE =
  'The backend restarted while this task was running, so the task was lost. ' +
  'In dev, any edit to a backend .py file triggers uvicorn --reload and kills ' +
  'running tasks. Re-run the operation.'

/** Poll a task until it reaches a terminal state.
 *
 * `fetchStatus` must reject with an Error; if the rejection carries a
 * `status` property it is treated as the HTTP status code.
 */
export async function pollTaskLoop(
  taskId: string,
  fetchStatus: () => Promise<TaskStatus>,
  opts: PollOptions = {},
): Promise<TaskStatus> {
  const interval = opts.intervalMs ?? 1000
  // Generous on purpose: a transient blip must not abandon a task that is
  // still running server-side. Only a 404 proves the task is gone.
  const maxTransient = opts.maxTransientFailures ?? 8
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)))

  let failures = 0
  while (true) {
    try {
      const status = await fetchStatus()
      failures = 0
      if (status.status !== 'running') {
        return status
      }
      opts.onProgress?.(status)
    } catch (err) {
      // A 404 is not a connection problem: the server answered and does not
      // know the task. The registry is in-process, so the process restarted
      // (or the entry expired). No retry can bring the task back.
      if ((err as { status?: number }).status === 404) {
        return { task_id: taskId, status: 'error', error: TASK_LOST_MESSAGE }
      }
      failures++
      if (failures >= maxTransient) {
        return {
          task_id: taskId,
          status: 'error',
          error: `Lost connection to task: ${(err as Error).message}`,
        }
      }
    }
    await sleep(interval)
  }
}
