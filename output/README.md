# output/ — full Kaggle output archives (Git LFS)

`output/paperNN_outputs.zip` is the **complete** output of a paper's Kaggle run: every model file (`best.pt` and
`last.pt`), logs, curves and predictions. Each is 40–250 MB, so these files are stored with **Git LFS**
(see [`.gitattributes`](../.gitattributes)).

| File | Member | Size |
|---|---|---|
| `paper03_outputs.zip` | Md. Yousuf | 41 MB |
| `paper11_outputs.zip` | Md Rakibul Hasan | 74 MB |
| `paper20_outputs.zip` | Md. Yousuf | 60 MB |
| `paper27_outputs.zip` | Md. Yousuf | 249 MB |
| `paper28_outputs.zip` | Md. Yousuf | 216 MB |

**Before you add one, read this:**

- GitHub Free gives this repo **10 GiB of LFS storage and 10 GiB of LFS downloads per month**. Every full clone
  downloads all of these zips (already about 650 MB). When the download quota runs out, LFS stops working for
  everyone until the next month.
- So the **team Drive (`498R / weights / paperNN /`) is the main place** for outputs zips. Rokib's papers
  02, 06, 08, 23 and 24 keep theirs on Drive only.
- Add a zip here with git on the command line (run `git lfs install` once): the web page cannot upload files over
  25 MB, and git refuses a normal (non-LFS) file over 100 MB.
- To clone without downloading these zips: `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Rokib-Hasan-Oli/CSE498R.git`
  and later `git lfs pull --include="output/paper27_outputs.zip"` for the one you need.
