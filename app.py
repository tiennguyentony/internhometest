from flask import Flask, render_template, request, redirect, url_for, jsonify
from train import train, construct, test
from queue import Queue
from threading import Thread
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
socketIO = SocketIO(app)
jobs = []

job_queue = Queue()


@app.route("/")
def home():
    worker_thread = Thread(target=worker)
    worker_thread.start()
    return render_template("home.html", jobs=jobs)


@app.route("/new", methods=["GET", "POST"])
def new_job():
    if request.method == "POST":

        lr = (
            float(request.form["learning_rate"])
            if request.form["learning_rate"]
            else 0.001
        )
        bs = int(request.form["batch_size"]) if request.form["batch_size"] else 64
        epochs = int(request.form["epochs"]) if request.form["epochs"] else 10
        dropout = float(request.form["dropout"]) if request.form["dropout"] else 0.5
        for job in jobs:
            if job["hyperparameters"] == {
                "learning_rate": lr,
                "batch_size": bs,
                "dropout": dropout,
                "epochs": epochs,
            }:

                return redirect(url_for("home", message="Job already exists"))

        job = {
            "id": len(jobs),
            "status": "queued",
            "hyperparameters": {
                "learning_rate": lr,
                "batch_size": bs,
                "dropout": dropout,
                "epochs": epochs,
            },
            "progress": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "run_time": 0,
        }

        jobs.append(job)
        job_queue.put(job)
        return redirect(url_for("home", message="Job added"))

    return render_template("new.html")


def worker():
    while job_queue.qsize() > 0:
        job = job_queue.get()
        lr = job["hyperparameters"]["learning_rate"]
        bs = job["hyperparameters"]["batch_size"]
        dropout = job["hyperparameters"]["dropout"]
        epochs = job["hyperparameters"]["epochs"]
        model, train_loader, test_loader, loss_fn, optimizer = construct(
            lr, bs, dropout_p=dropout
        )

        def callback(current_epoch, total_epochs):
            progress = current_epoch / total_epochs
            job["progress"] = progress * 100
            job["status"] = "Training"
            socketIO.emit(
                "training progress",
                {
                    "job_id": job["id"],
                    "progress": progress,
                    "status": job["status"],
                },
            )

        start_time = time.time()

        model = train(
            model,
            train_loader,
            epochs=epochs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            callback=callback,
        )
        job["status"] = "Completed"
        job["accuracy"], job["precision"], job["recall"] = test(model, test_loader)
        run_time = time.time() - start_time
        job["run_time"] = run_time
        socketIO.emit(
            "training Completed",
            {
                "job_id": job["id"],
                "status": job["status"],
                "accuracy": job["accuracy"],
                "precision": job["precision"],
                "recall": job["recall"],
                "run_time": run_time,
            },
        )
        job_queue.task_done()


if __name__ == "__main__":
    socketIO.run(app, debug=True)
