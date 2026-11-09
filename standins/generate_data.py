"""STAND-IN generator - only used when the official generate_data.py is missing. Not for submission."""
import csv, os, random
from datetime import datetime, timedelta
rng = random.Random(1)
os.makedirs("data/raw", exist_ok=True); os.makedirs("data/policies", exist_ok=True)
def w(n,h,rows):
    with open(f"data/raw/{n}","w",newline="") as f:
        x=csv.writer(f); x.writerow(h); x.writerows(rows)
cats=["Grocery","Dairy","Snacks","Personal Care","Electronics"]
prods=[(f"P{i+1:03d}",f"Item{i}",cats[i%5],rng.randint(20,1500)) for i in range(40)]
w("products.csv",["product_id","product_name","category","price_inr"],prods)
C=["Chennai","Mumbai","Bengaluru","Delhi","Hyderabad","Pune"]
cust=[]
for i in range(1200):
    cust.append([f"C{i+1:04d}",rng.choice([c.lower() if rng.random()<.2 else " "+c+" " if rng.random()<.1 else c for c in C]),
      (datetime(2024,1,1)+timedelta(days=rng.randint(0,455))).strftime("%Y-%m-%d"),
      "" if rng.random()<.05 else rng.choice([999,5,30,40,25,50]), rng.choice(["male","M","female","F",""])])
w("customers.csv",["customer_id","city","signup_date","age","gender"],cust)
rows=[];oid=100000
for c in cust:
    for _ in range(rng.randint(0,40)):
        oid+=1;p=rng.choice(prods);q=rng.randint(1,2);d=rng.choice([0,0,10,20])
        ts=datetime(2025,4,1)+timedelta(days=rng.randint(0,182),minutes=rng.randint(0,1439))
        st="cancelled" if rng.random()<.1 else "completed"
        amt=round(p[3]*q*(1-d/100),2); dl="" if st=="cancelled" or rng.random()<.2 else rng.randint(8,75)
        if st=="completed" and rng.random()<.01: amt=rng.choice(["N/A","error"])
        r=[f"ORD-{oid}",c[0],p[0],q,ts.strftime("%Y-%m-%d %H:%M:%S"),st,d,amt,dl]
        rows.append(r)
        if rng.random()<.05:
            e=list(r);e[4]=(ts-timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S");e[5]="pending";e[8]="";rows.append(e)
        if rng.random()<.01: rows.append(list(r))
w("orders.csv",["order_id","customer_id","product_id","quantity","order_ts","status","discount_pct","order_amount","delivery_minutes"],rows)
T=["I was charged twice for order {o}. email me at a@b.com","My order {o} has not arrived, call 9876543210","The app crashes on login","Unauthorized transaction on my card for {o}"]
tk=[]
for n in range(1,165):
    tk.append([f"T{n:04d}",rng.choice(cust)[0],"",rng.choice(T).format(o="ORD-100005"),["billing","delivery","technical","fraud"][n%4]])
for i in range(4): tk[i][3]=""
for i in range(4,8): tk[i][3]="word "*400
w("tickets.csv",["ticket_id","customer_id","order_id","text","true_category"],tk)
open("data/policies/refund_policy.txt","w").write("Refund Policy. A refund can be requested only for completed orders. "*8+"A refund above INR 1000 needs a human lead.\n")
open("data/policies/delivery_policy.txt","w").write("Delivery Policy. Standard delivery is promised within 45 minutes. "*8+"\n")
w("policy_eval.csv",["question","expected_doc","expected_fact"],[("Who must approve a refund above INR 1000?","refund_policy.txt","human lead"),("Who won the cricket world cup final?","NONE","")])
late=[r for r in rows if r[5]=="completed" and str(r[8]).isdigit() and int(r[8])>45 and str(r[7]).replace('.','').isdigit()]
big=next(r for r in late if float(r[7])>1000); small=next(r for r in late if float(r[7])<=1000)
ontime=next(r for r in rows if r[5]=="completed" and str(r[8]).isdigit() and int(r[8])<=45)
canc=next(r for r in rows if r[5]=="cancelled")
sc=[("S1",small,f"My order {small[0]} reached me very late, refund please","REFUND",float(small[7])),
    ("S2",small,f"Order {small[0]} was late again, refund please","REFUND",float(small[7])),
    ("S3",big,f"Order {big[0]} arrived far too late, refund please","ESCALATE",float(big[7])),
    ("S4",ontime,f"Please refund order {ontime[0]}, I did not like it","DENIED",0.0),
    ("S5",canc,f"I want my money back for order {canc[0]}.","DENIED",0.0),
    ("S6",small,"Refund for order ORD-999999 please.","NOT_FOUND",0.0),
    ("S7",small,f"[LOOP] Refund order {small[0]}.","FALLBACK_MAX_ITERATIONS",0.0)]
w("agent_scenarios.csv",["scenario_id","customer_id","message","expected_outcome","expected_refund_inr"],[(a,b[1],c,d,f"{e:.2f}") for a,b,c,d,e in sc])
print("STAND-IN data generated (NOT the official generate_data.py)")
