import { describe, it, expect, vi } from 'vitest'
import { pollTaskLoop, TASK_LOST_MESSAGE, TaskStatus } from './taskPolling'

const instant = () => Promise.resolve()

function running(progress?: number): TaskStatus {
  return { task_id: 't1', status: 'running', progress }
}

function completed(): TaskStatus {
  return { task_id: 't1', status: 'completed', result: { ok: 1 } }
}

/** A fetcher that plays back a script of outcomes, one per call. */
function scripted(script: Array<TaskStatus | Error>) {
  const calls = { count: 0 }
  const fetchStatus = () => {
    const step = script[Math.min(calls.count, script.length - 1)]
    calls.count++
    return step instanceof Error ? Promise.reject(step) : Promise.resolve(step)
  }
  return { fetchStatus, calls }
}

function httpError(status: number, message: string): Error {
  const err = new Error(message) as Error & { status?: number }
  err.status = status
  return err
}

describe('pollTaskLoop', () => {
  it('polls until terminal and reports progress along the way', async () => {
    const onProgress = vi.fn()
    const { fetchStatus } = scripted([running(0.2), running(0.6), completed()])
    const result = await pollTaskLoop('t1', fetchStatus, { sleep: instant, onProgress })
    expect(result.status).toBe('completed')
    expect(onProgress).toHaveBeenCalledTimes(2)
  })

  it('returns a cancelled status as-is', async () => {
    const { fetchStatus } = scripted([{ task_id: 't1', status: 'cancelled' }])
    const result = await pollTaskLoop('t1', fetchStatus, { sleep: instant })
    expect(result.status).toBe('cancelled')
  })

  it('a 404 means the task is gone for good: fails immediately, without retries', async () => {
    // After a backend restart (uvicorn --reload fires on any .py edit) the
    // in-memory task registry is empty, so every poll 404s forever. Retrying
    // only delays the honest answer.
    const sleep = vi.fn(instant)
    const { fetchStatus, calls } = scripted([httpError(404, 'Task not found')])
    const result = await pollTaskLoop('t1', fetchStatus, { sleep })
    expect(result.status).toBe('error')
    expect(result.error).toBe(TASK_LOST_MESSAGE)
    expect(calls.count).toBe(1)
    expect(sleep).not.toHaveBeenCalled()
  })

  it('tolerates well over three consecutive transient failures by default', async () => {
    // A brief network blip must not abandon a task that is still running
    // server-side; only a 404 proves the task is gone.
    const { fetchStatus } = scripted([
      httpError(500, 'Internal Server Error'),
      httpError(500, 'Internal Server Error'),
      httpError(500, 'Internal Server Error'),
      httpError(500, 'Internal Server Error'),
      httpError(500, 'Internal Server Error'),
      completed(),
    ])
    const result = await pollTaskLoop('t1', fetchStatus, { sleep: instant })
    expect(result.status).toBe('completed')
  })

  it('gives up after maxTransientFailures consecutive failures', async () => {
    const { fetchStatus, calls } = scripted([httpError(500, 'boom')])
    const result = await pollTaskLoop('t1', fetchStatus, {
      sleep: instant,
      maxTransientFailures: 2,
    })
    expect(result.status).toBe('error')
    expect(result.error).toBe('Lost connection to task: boom')
    expect(calls.count).toBe(2)
  })

  it('a successful poll resets the transient-failure budget', async () => {
    const { fetchStatus } = scripted([
      httpError(500, 'boom'),
      running(),
      httpError(500, 'boom'),
      completed(),
    ])
    const result = await pollTaskLoop('t1', fetchStatus, {
      sleep: instant,
      maxTransientFailures: 2,
    })
    expect(result.status).toBe('completed')
  })
})
