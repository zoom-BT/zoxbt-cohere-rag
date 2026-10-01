# zoxbt-cohere-rag

RAG over **[zoxbt.is-a.dev](https://zoxbt.is-a.dev)** - posts, projects, research, certifications.

**Author:** [Balbino Tchoutzine](https://github.com/zoom-BT)
**Content:**[ZoxBT_Blogfolio](https://github.com/zoom-BT/ZoxBT_Blogfolio)

Stack: **Cohere Embed + Rerank**, BM25 baseline, eval on labeled questions.

#Setup
```powersheell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# COHERE_API_KEY + BLOGFOLIO_PATH vers ton clone ZoxBT_Blogfolio
```
# Edit .env  -> COHERE_API_KEY (dashboard.cohere.com)
git clone --depth 1 https://github.com/zoom-BT/ZoxBT_Blogfolio.git _blogfolio_src 

# Pipeline
python -m zoxbt_cohere_rag ingest
python -m zoxbt_cohere_rag build-index
python -m zoxbt_cohere_rag eval --json
python -m zoxbt_cohere_rag ask "What is Gardienne?"

## Layout

| Path | Role |
|------|------|
| `zoxbt_cohere_rag/` | Package Python (`ingest`, `retrieve`, CLI) |
| `eval/questions.jsonl` | Jeu d’eval (question → `relevant_chunk_ids`) |
| `corpus_static/` | MD/About hors MDX Blogfolio (optionnel) |
| `data/` | `chunks.jsonl` + `index/` — **non versionné** |
| `_blogfolio_src/` | Clone [ZoxBT_Blogfolio](https://github.com/zoom-BT/ZoxBT_Blogfolio) **ou** chemin via `.env` |

## Methods (`ask --method`)

| Method | Description |
|--------|-------------|
| `bm25` | Recherche par mots-clés (baseline, sans embed query) |
| `embed` | Similarité sur embeddings Cohere (`embed-multilingual-v3.0`) |
| `embed_rerank` | Top ~15 en embed → **Rerank** Cohere → top 5 (**défaut** pour `ask`) |

Exemples :

```powershell
python -m zoxbt_cohere_rag ask "What is Gardienne?"
python -m zoxbt_cohere_rag ask "Rendements agricoles Tchad Zindi" --method bm25
python -m zoxbt_cohere_rag ask "Qu'est-ce que Gardienne ?" --method embed
python -m zoxbt_cohere_rag ask "What is Gardienne?" --method embed_rerank


---

**Ordre complet suggéré dans le README :**

1. Titre + liens site / Blogfolio  
2. Setup  
3. Pipeline  
4. **Layout** (arbre ou tableau)  
5. **Methods + ask** (ci-dessus)  
6. **Eval**  
7. **CLI récap**  
8. **`.env`**  
9. **Mise à jour**  
10. **Intégration site** + **License**

Ensuite : `git add README.md` → commit → push.