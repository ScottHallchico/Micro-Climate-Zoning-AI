import numpy as np

class CFDDispatcher:
    def __init__(self):
        self.job_queue = []
        self.completed_jobs = []
        
    def dispatch(self, job_id, coords, ws, wd):
        job = {
            "job_id": job_id,
            "status": "QUEUED",
            "coords_len": len(coords),
            "ws": ws,
            "wd": wd,
            "provenance": "automated_fallback"
        }
        self.job_queue.append(job)
        return job_id
        
    def execute_queue(self):
        for job in self.job_queue:
            job["status"] = "COMPLETED"
            # Mock CFD completion
            job["result"] = {
                "wsi": np.random.uniform(0.3, 0.7),
                "corridor_score": np.random.uniform(0.1, 0.5)
            }
            self.completed_jobs.append(job)
        self.job_queue = []
        
    def get_job_status(self, job_id):
        for job in self.completed_jobs:
            if job["job_id"] == job_id:
                return job
        for job in self.job_queue:
            if job["job_id"] == job_id:
                return job
        return None
