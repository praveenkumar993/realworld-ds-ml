import sqlite3
import pandas as pd

conn = sqlite3.connect(
    r"C:\Users\dsp96\Desktop\realworld-ds-ml\finance\insurance_claim_prediction\data\insurance.db"
)

# Validate NCB vs claim rate — must be monotonically decreasing
print("NCB vs Claim Rate:")
pd.read_sql_query("""
    SELECT
        p.ncb_percentage,
        COUNT(*)                               AS policies,
        ROUND(AVG(l.has_claim)*100, 2)        AS claim_rate_pct,
        ROUND(AVG(CASE WHEN l.claim_amount > 0
                       THEN l.claim_amount END), 0) AS avg_claim_amt
    FROM claim_labels l
    JOIN policies pol ON l.policy_id = pol.policy_id
    JOIN policyholders p ON pol.policyholder_id = p.policyholder_id
    WHERE p.ncb_percentage IS NOT NULL
    GROUP BY p.ncb_percentage
    ORDER BY p.ncb_percentage
""", conn).pipe(print)

# Validate vehicle age vs claim rate — must increase with age
print("\nVehicle Age vs Claim Rate:")
pd.read_sql_query("""
    SELECT
        v.vehicle_age_years,
        COUNT(*)                               AS policies,
        ROUND(AVG(l.has_claim)*100, 2)        AS claim_rate_pct
    FROM claim_labels l
    JOIN policies pol ON l.policy_id = pol.policy_id
    JOIN vehicles v ON pol.policyholder_id = v.policyholder_id
    WHERE v.vehicle_age_years IS NOT NULL
    GROUP BY v.vehicle_age_years
    ORDER BY v.vehicle_age_years
""", conn).pipe(print)

conn.close()