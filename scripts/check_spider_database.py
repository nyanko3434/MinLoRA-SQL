import json, sqlite3

dev = json.load(open("data/spider_data/dev.json"))
print("dev examples:", len(dev))

ex = dev[0]
db_id = ex["db_id"]
con = sqlite3.connect(f"data/spider_data/database/{db_id}/{db_id}.sqlite")
cur = con.cursor()

print("question:", ex["question"])
print("gold sql:", ex["query"])
print("result:", cur.execute(ex["query"]).fetchall()[:5])

# schema check
tables = json.load(open("data/spider_data/tables.json"))
schema = next(t for t in tables if t["db_id"] == db_id)
print("tables:", schema["table_names_original"])