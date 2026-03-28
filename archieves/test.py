from sheets import read_tab

rows = sorted(
    [r for r in read_tab("nepse_indices") if r.get("current_value") and r.get("index_id") == "58"],
    key=lambda x: x.get("date", "")
)
breadth_rows=sorted(
            [r for r in read_tab("market_breadth") 
             if r.get("date") 
             and str(r.get("advancing", "")).strip() 
             and str(r.get("declining", "")).strip()],
            key=lambda x: x.get("date", ""),
            reverse=True

)
# for br in breadth_rows:
#     br_date = str(br.get("date", "")).strip()
#     if br_date and br_date <= latest_nepse_date:
#         recent_breadth = br
#         print("Selected breadth date: %s (NEPSE date: %s)", br_date, latest_nepse_date)
#         break

print(rows)[-1]

