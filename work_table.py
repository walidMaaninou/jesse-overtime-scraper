import streamlit as st
import os
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
import base64
import re
from PIL import Image, ImageFilter, ImageOps
from pdf2image import convert_from_bytes
import pytesseract
import openai

# ---------------- API key input ---------------- #
openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
if not openai_api_key:
    st.stop()
openai.api_key = openai_api_key

# ---------------- Original functions (unchanged) ---------------- #
def extract_times(text):
    tokens = re.split(r'[ .|;,-]+', text)
    times = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if re.match(r'^(\d{1,2})(\d{2})$', token):
            times.append(f"{int(token[:-2])}:{token[-2:]}")
        elif re.match(r'^\d{1,2}[:.]\d{2}$', token):
            times.append(token.replace('.', ':'))
        elif re.match(r'^\d{1,2}[:.]\d{2}\s?[APMapm]{2}$', token):
            times.append(token.replace('.', ':').upper())
    return times

def process_line(line):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day in days:
        if day in line:
            times = extract_times(line)[-5:]
            if len(times) != 5:
                return None
            return [day] + times
    return None

def extract_and_expand_date_range(text):
    pattern = r'(\d{1,2}/\d{1,2}/\d{4})\s*-\s*(\d{1,2}/\d{1,2}/\d{4})'
    match = re.search(pattern, text)
    if not match:
        return []
    start_str, end_str = match.groups()
    start_date = datetime.strptime(start_str, '%m/%d/%Y')
    end_date = datetime.strptime(end_str, '%m/%d/%Y')
    date_list = []
    current = start_date
    while current <= end_date:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list

def preprocess_image(image):
    image = image.convert("L")
    image = image.filter(ImageFilter.MedianFilter())
    image = ImageOps.invert(image)
    image = image.point(lambda x: 0 if x < 180 else 255, '1')
    image = image.resize((image.width * 2, image.height * 2))
    return image

def read_pdf_with_ocr_openai_bytes(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", 'Sun']
    output = []
    dates = []
    lines = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_b64 = base64.b64encode(buffered.read()).decode()
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "Extract the text from this scanned table. Keep original layout and order. Focus on text inside cells."},
                {"role": "user",
                 "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}]},
            ],
            max_tokens=2000,
            temperature=0
        )
        text = response.choices[0].message.content
        text_dates = pytesseract.image_to_string(image)
        if not dates:
            dates = extract_and_expand_date_range(text_dates)
        for line in text.splitlines():
            if any(day in line for day in days):
                lines.append(line)
    for line, date in zip(lines, dates):
        processed_line = process_line(line)
        print(line)
        print(processed_line)
        if processed_line:
            output.append([date] + processed_line)
    return output

def to_datetime_inferred(date_str, time_str, previous_time=None, is_first=False):
    try:
        time_str = time_str.replace('.', ':').strip()
        h, m = map(int, time_str.split(':'))
        dt = datetime.strptime(f"{date_str} {h}:{m}", "%Y-%m-%d %H:%M")
        if is_first and h < 8:
            dt += timedelta(hours=12)
        if previous_time and dt <= previous_time:
            dt += timedelta(hours=12)
        return dt
    except Exception:
        return None

def is_within_any(actual_start, actual_end, planned_intervals):
    uncovered = []
    current = actual_start
    for p_start, p_end in planned_intervals:
        if current >= actual_end:
            break
        if p_end <= current or p_start >= actual_end:
            continue
        if p_start > current:
            uncovered.append((current, min(p_start, actual_end)))
        current = max(current, p_end)
    if current < actual_end:
        uncovered.append((current, actual_end))
    return uncovered

def compute_overtime_intervals(planned_rows, df_actual):
    results = []
    for row in planned_rows:
        date, day, in1, out1, in2, out2, shift = row
        t1 = to_datetime_inferred(date, in1, is_first=True)
        t2 = to_datetime_inferred(date, out1, previous_time=t1)
        t3 = to_datetime_inferred(date, in2, previous_time=t2)
        t4 = to_datetime_inferred(date, out2, previous_time=t3)
        planned_intervals = [(t1, t2), (t3, t4)]
        actual_intervals = df_actual[df_actual['Day'] == date][['Shift Start', 'Shift End']]
        actual_intervals = list(actual_intervals.itertuples(index=False, name=None))
        for a_start, a_end in actual_intervals:
            uncovered_parts = is_within_any(a_start, a_end, planned_intervals)
            for r_start, r_end in uncovered_parts:
                overtime_duration = int((r_end - r_start).total_seconds() // 60)
                if overtime_duration:
                    results.append({
                        "Day": date,
                        "Overtime Start": r_start,
                        "Overtime End": r_end,
                        "Overtime Duration (min)": overtime_duration
                    })
    return pd.DataFrame(results)

# ---------------- Minimal Streamlit UI ---------------- #
st.title("PDF + CSV Overtime Extractor")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
csv_file = st.file_uploader("Upload CSV", type=["csv"])

if pdf_file and csv_file and st.button("Run"):
    with st.status("Processing…", expanded=True) as status:
        try:
            status.write("1/4 Reading CSV…")
            df = pd.read_csv(csv_file)
            df['Decision Time'] = pd.to_datetime(df['Moderator Decisions Decision Made At (EST) Time'])
            df = df.sort_values('Decision Time').reset_index(drop=True)

            status.write("2/4 Computing shifts from CSV…")
            shifts = []
            current_shift_start = df.loc[0, 'Decision Time']
            current_day = current_shift_start.date()

            for i in range(1, len(df)):
                now = df.loc[i, 'Decision Time']
                prev = df.loc[i - 1, 'Decision Time']
                if (now - prev) >= timedelta(minutes=10):
                    shifts.append({'Day': current_day, 'Shift Start': current_shift_start, 'Shift End': prev})
                    current_shift_start = now
                    current_day = now.date()

            shifts.append({'Day': current_day, 'Shift Start': current_shift_start, 'Shift End': df.loc[len(df) - 1, 'Decision Time']})
            shift_df = pd.DataFrame(shifts)
            print(shift_df)
            status.write("3/4 OCR on PDF to extract planned rows…")
            planned_rows = read_pdf_with_ocr_openai_bytes(pdf_file.read())
            print(planned_rows)
            status.write("4/4 Computing overtime…")
            shift_df['Day'] = shift_df['Day'].astype(str)
            shift_df['Shift Start'] = pd.to_datetime(shift_df['Shift Start'])
            shift_df['Shift End'] = pd.to_datetime(shift_df['Shift End'])
            overtime_df = compute_overtime_intervals(planned_rows, shift_df)

            status.update(label="Done ✅", state="complete")
        except Exception as e:
            status.update(label="Failed ❌", state="error")
            st.exception(e)
            st.stop()

    # Show tables
    st.subheader("Overtime")
    st.dataframe(overtime_df, use_container_width=True, hide_index=True)

    st.subheader("Overtime Aggregation")
    if not overtime_df.empty:
        overtime_agg = (
            overtime_df.groupby('Day', as_index=False)['Overtime Duration (min)'].sum()
            .rename(columns={'Overtime Duration (min)': 'Total Overtime (min)'})
        )
        st.dataframe(overtime_agg, use_container_width=True, hide_index=True)
    else:
        st.info("No overtime intervals detected to aggregate.")

    # Keep your existing CSV download (optional)
    csv_data = overtime_df.to_csv(index=False).encode()
    st.download_button("Download Overtime CSV", csv_data, "overtime.csv", "text/csv")
