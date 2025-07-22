# AIG-chatbot-Documentation
In my AIG role, the workflow centered on building a scalable conversational AI: data ingestion from S3/DynamoDB, preprocessing and fine-tuning LLMs with RAG, validation via k-fold CV, optimization through pruning/dashboards, and deployment as microservices with CI/CD. This integrated AWS Lambda for triggers, SageMaker for training, and FastAPI for APIs, handling 10K+ queries/month with 95% F1 and 27% latency cut.

Brief Workflow:

Ingest/Preprocess: Load customer data, clean with Pandas, generate embeddings.
Fine-Tune: Use Hugging Face for BERT/GPT-3, LangChain for prompts, CV on SageMaker.
Optimize/Validate: Prune models, monitor latency with Tableau/SQL aggregates.
Deploy: FastAPI async APIs, Jenkins/Docker CI/CD.
