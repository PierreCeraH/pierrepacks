from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict")
def predict(worst_status_active_inv,
            account_worst_status_12_24m,
            account_worst_status_6_12m,
            account_incoming_debt_vs_paid_0_24m,
            account_worst_status_3_6m,
            account_status,
            account_worst_status_0_3m,
            avg_payment_span_0_3m,
            avg_payment_span_0_12m,
            num_active_div_by_paid_inv_0_12m,
            num_arch_written_off_0_12m,
            num_arch_written_off_12_24m,
            account_days_in_term_12_24m,
            account_days_in_rem_12_24m,
            account_days_in_dc_12_24m,
            sum_paid_inv_0_12m,
            sum_capital_paid_account_12_24m,
            sum_capital_paid_account_0_12m,
            recovery_debt,
            status_max_archived_0_12_months,
            status_max_archived_0_6_months,
            status_3rd_last_archived_0_24m,
            status_2nd_last_archived_0_24m,
            status_last_archived_0_24m,
            num_unpaid_bills,
            time_hours,
            status_max_archived_0_24_months,
            num_arch_ok_12_24m,
            num_arch_rem_0_12m,
            num_arch_ok_0_12m,
            num_arch_dc_12_24m,
            num_arch_dc_0_12m,
            num_active_inv,
            default,
            name_in_email,
            max_paid_inv_0_24m,
            max_paid_inv_0_12m,
            has_paid,
            merchant_group,
            merchant_category,
            account_amount_added_12_24m,
            age,
            uuid):

    X_pred=pd.DataFrame({'worst_status_active_inv' : worst_status_active_inv,
            'account_worst_status_12_24m' : account_worst_status_12_24m,
            'account_worst_status_6_12m' : account_worst_status_6_12m,
            'account_incoming_debt_vs_paid_0_24m' : account_incoming_debt_vs_paid_0_24m,
            'account_worst_status_3_6m' : account_worst_status_3_6m,
            'account_status' : account_status,
            'account_worst_status_0_3m' : account_worst_status_0_3m,
            'avg_payment_span_0_3m' : avg_payment_span_0_3m,
            'avg_payment_span_0_12m' : avg_payment_span_0_12m,
            'num_active_div_by_paid_inv_0_12m' : num_active_div_by_paid_inv_0_12m,
            'num_arch_written_off_0_12m' : num_arch_written_off_0_12m,
            'num_arch_written_off_12_24m' : num_arch_written_off_12_24m,
            'account_days_in_term_12_24m' : account_days_in_term_12_24m,
            'account_days_in_rem_12_24m' : account_days_in_rem_12_24m,
            'account_days_in_dc_12_24m' : account_days_in_dc_12_24m,
            'sum_paid_inv_0_12m' : sum_paid_inv_0_12m,
            'sum_capital_paid_account_12_24m' : sum_capital_paid_account_12_24m,
            'sum_capital_paid_account_0_12m' : sum_capital_paid_account_0_12m,
            'recovery_debt' : recovery_debt,
            'status_max_archived_0_12_months' : status_max_archived_0_12_months,
            'status_max_archived_0_6_months' : status_max_archived_0_6_months,
            'status_3rd_last_archived_0_24m' : status_3rd_last_archived_0_24m,
            'status_2nd_last_archived_0_24m' : status_2nd_last_archived_0_24m,
            'status_last_archived_0_24m' : status_last_archived_0_24m,
            'num_unpaid_bills' : num_unpaid_bills,
            'time_hours' : time_hours,
            'status_max_archived_0_24_months' : status_max_archived_0_24_months,
            'num_arch_ok_12_24m' : num_arch_ok_12_24m,
            'num_arch_rem_0_12m' : num_arch_rem_0_12m,
            'num_arch_ok_0_12m' : num_arch_ok_0_12m,
            'num_arch_dc_12_24m' : num_arch_dc_12_24m,
            'num_arch_dc_0_12m' : num_arch_dc_0_12m,
            'num_active_inv' : num_active_inv,
            'default' : default,
            'name_in_email' : name_in_email,
            'max_paid_inv_0_24m' : max_paid_inv_0_24m,
            'max_paid_inv_0_12m' : max_paid_inv_0_12m,
            'has_paid' : has_paid,
            'merchant_group' : merchant_group,
            'merchant_category' : merchant_category,
            'account_amount_added_12_24m' : account_amount_added_12_24m,
            'age' : age,
            'uuid' : uuid},index=[0])


    ignore_cols = ['name_in_email','uuid']
    cols = list(X_pred.columns)

    for col in cols :
        if col not in ignore_cols :
            X_pred[col] = X_pred[col].astype(float)

    pipeline = joblib.load('model.joblib')

    prediction = pipeline.predict(X_pred)

    return f'Prediction de bibi : {prediction}'
