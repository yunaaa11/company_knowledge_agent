# import chromadb

# # 改成你项目实际的路径
# client = chromadb.PersistentClient(path="./vector_db")

# collections = client.list_collections()
# print(f"所有集合：{collections}")

# for col_info in collections:
#     col = client.get_collection(col_info.name)
#     count = col.count()
#     print(f"\n集合名: {col_info.name}，共 {count} 条")
    
#     if count > 0:
#         results = col.get(include=["documents", "metadatas"])
#         for doc, meta in zip(results["documents"][:5], results["metadatas"][:5]):
#             print(f"  来源: {meta}")
#             print(f"  内容: {doc[:80]}")
#             print("  ---")
# check2.py
# import chromadb

# client = chromadb.PersistentClient(path="./vector_db")
# col = client.get_collection("enterprise_paper")

# # 搜索和"鼠标"相关的所有chunk，看内容
# results = col.get(include=["documents", "metadatas"])
# for doc, meta in zip(results["documents"], results["metadatas"]):
#     if "鼠标" in doc or "耐用品" in doc or "办公用品" in doc:
#         print(f"来源: {meta['source']}")
#         print(f"内容: {doc[:200]}")
#         print("---")
import chromadb

client = chromadb.PersistentClient(path="./vector_db")
col = client.get_collection("enterprise_paper")

results = col.get(include=["documents", "metadatas"])
print(f"总共 {len(results['documents'])} 条\n")

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"[{i}] 来源: {meta['source']}")
    print(f"     内容前50字: {doc[:50]}")
    print()