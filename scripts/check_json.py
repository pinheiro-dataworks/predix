import json

with open("dashboard/data/dashboard_data.json") as f:
    d = json.load(f)

for month in ["all", "october", "november"]:
    de = d["overview"][month]["daily_events"]
    assert len(de["dates"]) > 0, f"Overview {month} empty"
    assert len(d["products"][month]["treemap"]["labels"]) > 0
    assert len(d["behavior"][month]["heatmap"]["z"]) > 0

ml = d["ml"]
assert set(ml["roc_curves"]) == {
    "Recency baseline", "Logistic Regression", "Random Forest", "Gradient Boosting",
}
assert len(ml["feature_importance"]["features"]) == 13
assert all(0.5 <= v <= 1 for v in ml["model_pr_auc"].values())
assert len(ml["affinity_matrix"]["z"]) == 8

ch = d["churn"]
assert len(ch["rfm_3d"]["r"]) == 3000
assert len(ch["segments"]["names"]) >= 4
assert ch["eligible"] == ch["retained"] + ch["churned"]
assert {"as_of", "horizon_days"} <= ch.keys()

print("ALL CHECKS PASSED")
for m in ["all", "october", "november"]:
    k = d["overview"][m]["kpis"]
    print(f"  {m}: revenue={k['total_revenue']:,.0f}  orders={k['total_orders']:,}")
print(f"Churn rate: {ch['churn_rate']:.1f}% (as_of={ch['as_of']}, N={ch['horizon_days']}d)")
print(f"Segments: {ch['segments']['names']}")
print(f"GB AUC: {d['ml']['gb_auc']:.4f}")
