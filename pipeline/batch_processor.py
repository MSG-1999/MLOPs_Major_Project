# pipeline/batch_processor.py
"""
Batch Processor: Queue-based system for processing multiple generation jobs.
Supports priority queuing, job status tracking, and concurrent execution.
"""

import time
import uuid
import threading
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue, Empty
from typing import Optional, List, Callable, Dict, Any

from loguru import logger
from PIL import Image

from pipeline.inference_engine import InferenceEngine, GenerationConfig, GenerationResult


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class BatchJob:
    """A single job in the batch queue."""
    priority: int                          # Lower = higher priority
    job_id: str = field(compare=False)
    config: GenerationConfig = field(compare=False)
    status: JobStatus = field(default=JobStatus.QUEUED, compare=False)
    result: Optional[GenerationResult] = field(default=None, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    started_at: Optional[float] = field(default=None, compare=False)
    completed_at: Optional[float] = field(default=None, compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)

    @property
    def wait_time_s(self) -> float:
        if self.started_at:
            return self.started_at - self.created_at
        return time.time() - self.created_at

    @property
    def run_time_s(self) -> Optional[float]:
        if self.result:
            return self.result.generation_time_s
        return None


class BatchProcessor:
    """
    Processes generation jobs from a priority queue.
    Runs a background worker thread for continuous job processing.

    Usage:
        processor = BatchProcessor(engine)
        processor.start()

        job_id = processor.submit(config, priority=1)
        status = processor.get_status(job_id)

        processor.stop()
    """

    def __init__(
        self,
        engine: InferenceEngine,
        max_queue_size: int = 50,
    ):
        self.engine = engine
        self.queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        self._jobs: Dict[str, BatchJob] = {}
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._total_processed = 0
        self._total_failed = 0

    def start(self):
        """Start the background worker thread."""
        if self._running:
            logger.warning("BatchProcessor already running")
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="BatchProcessor-Worker",
        )
        self._worker_thread.start()
        logger.info("BatchProcessor worker started")

    def stop(self, wait: bool = True, timeout: float = 30.0):
        """Stop the worker thread."""
        self._running = False
        if wait and self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        logger.info("BatchProcessor stopped")

    def submit(
        self,
        config: GenerationConfig,
        priority: int = 5,
        callback: Optional[Callable[[GenerationResult], None]] = None,
    ) -> str:
        """
        Submit a job to the queue.

        Args:
            config: Generation configuration
            priority: Job priority (1=highest, 10=lowest)
            callback: Optional function called with result when complete

        Returns:
            job_id string
        """
        job_id = str(uuid.uuid4())[:12]
        job = BatchJob(
            priority=priority,
            job_id=job_id,
            config=config,
            callback=callback,
        )

        with self._lock:
            self._jobs[job_id] = job

        self.queue.put(job)
        logger.info(f"Job {job_id} queued (priority={priority}, queue_size={self.queue.qsize()})")
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job (cannot cancel running jobs)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                logger.info(f"Job {job_id} cancelled")
                return True
        return False

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status and result for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "priority": job.priority,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "wait_time_s": job.wait_time_s,
            "run_time_s": job.run_time_s,
            "result": job.result,
            "error": job.result.error if job.result else None,
        }

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get overall queue statistics."""
        with self._lock:
            jobs = list(self._jobs.values())

        by_status = {}
        for s in JobStatus:
            by_status[s.value] = sum(1 for j in jobs if j.status == s)

        return {
            "queue_depth": self.queue.qsize(),
            "total_jobs": len(jobs),
            "total_processed": self._total_processed,
            "total_failed": self._total_failed,
            "by_status": by_status,
            "worker_running": self._running,
        }

    def get_completed_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently completed job results."""
        with self._lock:
            completed = [
                j for j in self._jobs.values()
                if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            ]
        completed.sort(key=lambda j: j.completed_at or 0, reverse=True)
        return [self.get_status(j.job_id) for j in completed[:limit]]

    def _worker_loop(self):
        """Main worker loop — runs in background thread."""
        logger.info("Worker loop started")
        while self._running:
            try:
                job: BatchJob = self.queue.get(timeout=1.0)
            except Empty:
                continue

            # Skip cancelled jobs
            if job.status == JobStatus.CANCELLED:
                self.queue.task_done()
                continue

            # Process job
            with self._lock:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

            logger.info(f"Processing job {job.job_id}")

            try:
                result = self.engine.generate(job.config)
                with self._lock:
                    job.result = result
                    job.completed_at = time.time()
                    job.status = JobStatus.COMPLETED if result.success else JobStatus.FAILED

                    if result.success:
                        self._total_processed += 1
                    else:
                        self._total_failed += 1

                # Fire callback
                if job.callback:
                    try:
                        job.callback(result)
                    except Exception as e:
                        logger.warning(f"Job callback failed: {e}")

                logger.info(
                    f"Job {job.job_id} {'completed' if result.success else 'failed'} "
                    f"in {result.generation_time_s:.1f}s"
                )

            except Exception as e:
                logger.error(f"Worker error for job {job.job_id}: {e}")
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.completed_at = time.time()
                    self._total_failed += 1
            finally:
                self.queue.task_done()

        logger.info("Worker loop stopped")
