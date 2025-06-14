# 🗃️ DVC (Data Version Control) Notes

## 🔍 What is DVC?

DVC (Data Version Control) is an open-source version control system for machine learning projects. It extends Git capabilities to handle large files, datasets, and machine learning models.

---

## 🎯 Key Features

* Version control for data, models, and intermediate files.
* Git-friendly: uses `.dvc` files to track large files.
* Data pipeline support with reproducibility.
* Remote storage integration (S3, GCS, Azure, SSH, etc.).
* Metrics tracking and experiment comparisons.

---

## ⚙️ How DVC Works

* Large files are replaced with small `.dvc` metafiles in Git.
* Actual data is stored in a separate DVC cache or remote storage.
* Use `dvc push` and `dvc pull` to sync data with remote storage.

---

## 🧰 Most Useful Commands

| Command                          | Description                                |
| -------------------------------- | ------------------------------------------ |
| `dvc init`                       | Initialize a DVC project                   |
| `dvc add <file>`                 | Track a data file or model with DVC        |
| `dvc status`                     | Show which files have changed              |
| `dvc push` / `dvc pull`          | Sync data with remote storage              |
| `dvc remote add -d <name> <url>` | Add remote storage location                |
| `dvc repro`                      | Re-run stages that depend on changed files |
| `dvc stage add`                  | Define pipeline stages in `dvc.yaml`       |
| `dvc dag`                        | Visualize pipeline as a DAG                |
| `dvc metrics show`               | Show current metrics values                |
| `dvc exp run`                    | Run an experiment variant                  |
| `dvc exp show`                   | Show all experiments and metrics           |
| `dvc exp diff`                   | Compare two experiments                    |
| `dvc gc`                         | Clean unused cache and outputs             |

---

## 📦 Pipelines with `dvc.yaml`

DVC allows you to define pipelines via YAML:

```yaml
deps:
  - data/raw.csv
outs:
  - data/processed.csv
cmd: python process.py
```

Use:

```bash
dvc stage add -n process -d data/raw.csv -o data/processed.csv python process.py
dvc repro
```

---

## 🌐 Remote Storage Options

* Amazon S3
* Google Cloud Storage
* Azure Blob Storage
* SSH, WebDAV, SFTP

Add remote:

```bash
dvc remote add -d myremote s3://mybucket/data
```

---

## 📊 Metrics and Plots

Track and compare metrics using DVC:

```bash
dvc metrics show
dvc exp show
dvc exp diff
```

---

## 🧪 Experiment Tracking

* Create and run experiments:

```bash
dvc exp run
```

* Compare and apply experiments:

```bash
dvc exp show
dvc exp diff
dvc exp apply <exp>
dvc commit
```

---

## 🧠 Best Practices

* Commit `.dvc`, `.dvc/config`, and `dvc.yaml` files to Git.
* Use `.gitignore` for large data files.
* Set up CI/CD to automate `dvc repro` and metric collection.

---

## ✅ Benefits

* Reproducibility
* Collaboration
* Scalability
* Lightweight and Git-native

---

## 🔚 Summary

DVC helps manage data, models, and ML workflows efficiently by bringing software engineering practices into ML projects. It works alongside Git to ensure end-to-end reproducibility and collaboration.
