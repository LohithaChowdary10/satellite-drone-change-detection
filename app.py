import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from datetime import datetime
import io
import html

st.set_page_config(page_title="GeoChange | Disaster Monitoring", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
.stApp {background:#f4f7fb;}
.block-container {padding-top:1.5rem; max-width:1450px;}
.hero {background:linear-gradient(110deg,#102a43,#176b87);padding:25px 30px;border-radius:18px;color:white;margin-bottom:18px;}
.hero h1 {color:white;margin:0;font-size:30px;}
.hero p {color:#d9edf5;margin:7px 0 0 0;}
.metric {background:white;border:1px solid #e5eaf0;padding:17px 19px;border-radius:13px;min-height:105px;}
.metric small {color:#607080;font-size:13px;}
.metric h2 {margin:6px 0 0 0;color:#102a43;font-size:27px;}
.panel {background:white;border:1px solid #e5eaf0;padding:17px;border-radius:13px;}
.note {background:#fff8e6;border-left:4px solid #e0a526;padding:12px 15px;border-radius:7px;color:#604b1b;}
</style>
""", unsafe_allow_html=True)

def make_sample_pair():
    """Create a reproducible synthetic aerial scene and a changed version for testing."""
    rng=np.random.default_rng(7)
    h,w=420,640
    base=np.zeros((h,w,3),dtype=np.uint8)
    base[:]=(75,112,74)
    noise=rng.normal(0,10,(h,w,1))
    base=np.clip(base.astype(float)+noise,0,255).astype(np.uint8)
    # fields/roads/water/buildings
    cv2.rectangle(base,(25,30),(260,180),(80,135,85),-1)
    cv2.rectangle(base,(300,35),(610,180),(65,125,75),-1)
    cv2.line(base,(0,260),(640,220),(155,150,130),18)
    cv2.rectangle(base,(400,275),(610,405),(135,105,75),-1)
    for x,y in [(90,235),(145,230),(210,300),(350,310),(520,230),(560,350)]:
        cv2.rectangle(base,(x,y),(x+32,y+25),(185,180,165),-1)
    cv2.circle(base,(315,115),45,(120,145,65),-1)
    post=base.copy()
    # simulated flood region and vegetation loss
    cv2.fillPoly(post,[np.array([[35,45],[245,45],[275,145],[220,205],[70,180]],np.int32)],(155,95,45))
    cv2.rectangle(post,(390,285),(615,410),(155,95,45),-1)
    cv2.rectangle(post,(80,220),(180,265),(65,75,70),-1)
    return Image.fromarray(cv2.cvtColor(base,cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(post,cv2.COLOR_BGR2RGB))

def read_image(upload):
    image=Image.open(upload).convert("RGB")
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def align_images(reference, moving):
    """ECC alignment; returns moving image aligned to reference when successful."""
    try:
        ref_gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
        mov_gray=cv2.cvtColor(moving,cv2.COLOR_BGR2GRAY)
        warp=np.eye(2,3,dtype=np.float32)
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,100,1e-6)
        cv2.findTransformECC(ref_gray,mov_gray,warp,cv2.MOTION_AFFINE,criteria)
        h,w=reference.shape[:2]
        aligned=cv2.warpAffine(moving,warp,(w,h),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)
        return aligned, True
    except cv2.error:
        return moving, False

def analyze(before, after, threshold_value, min_area):
    h,w=before.shape[:2]
    after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
    aligned_after, aligned_ok=align_images(before,after)
    # compare the reference image with the aligned post-event image
    lab1=cv2.cvtColor(before,cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2=cv2.cvtColor(aligned_after,cv2.COLOR_BGR2LAB).astype(np.float32)
    diff=np.linalg.norm(lab1-lab2,axis=2)
    diff8=np.uint8(np.clip(diff*2.0,0,255))
    blur=cv2.GaussianBlur(diff8,(5,5),0)
    if threshold_value == 0:
        _,mask=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _,mask=cv2.threshold(blur,int(threshold_value),255,cv2.THRESH_BINARY)
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    clean=np.zeros_like(mask)
    regions=[]
    for i in range(1,count):
        x,y,bw,bh,area=stats[i]
        if area>=min_area:
            clean[labels==i]=255
            regions.append({"Region":len(regions)+1,"X":int(x),"Y":int(y),
                            "Width":int(bw),"Height":int(bh),"Area (pixels)":int(area)})
    changed=int(np.count_nonzero(clean))
    total=h*w
    pct=changed/total*100
    severity="Low" if pct<5 else ("Medium" if pct<15 else "High")
    overlay=aligned_after.copy()
    red=np.zeros_like(aligned_after); red[:]=(0,0,255)
    overlay=np.where(clean[:,:,None]>0,cv2.addWeighted(aligned_after,0.45,red,0.55,0),aligned_after)
    heat=cv2.applyColorMap(diff8,cv2.COLORMAP_JET)
    heat=cv2.addWeighted(aligned_after,0.55,heat,0.45,0)
    return before,aligned_after,clean,overlay,heat,pct,severity,regions,aligned_ok

def rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

st.markdown("""
<div class="hero">
  <h1>🛰️ GeoChange — Satellite & Drone Change Detection</h1>
  <p>Computer Vision platform for environmental monitoring and disaster impact assessment</p>
</div>
""",unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.caption("Configure the image comparison")
    threshold=st.slider("Change sensitivity",0,100,0,help="0 uses automatic Otsu thresholding. Higher values detect stronger differences only.")
    min_area=st.slider("Minimum region size (pixels)",10,2000,100,step=10)
    category=st.selectbox("Monitoring category",["General environmental change","Flood impact","Vegetation loss","Building / infrastructure damage","Landslide impact"])
    location=st.text_input("Location / study area", "Sample study area")
    st.divider()
    st.caption("Tip: use two images of the same place, captured at different times and from a similar angle.")

tab1,tab2,tab3=st.tabs(["📊 Dashboard & Analysis","🗺️ Change Visualization","📄 Report & Methodology"])

with tab1:
    st.subheader("Image collection and preprocessing")
    c1,c2=st.columns(2)
    with c1:
        before_file=st.file_uploader("Upload BEFORE image",type=["png","jpg","jpeg","tif","tiff"],key="before")
    with c2:
        after_file=st.file_uploader("Upload AFTER image",type=["png","jpg","jpeg","tif","tiff"],key="after")
    use_sample=st.button("Load sample images (for first-time testing)",type="secondary")
    if use_sample:
        s1,s2=make_sample_pair()
        st.session_state["sample_before"]=s1
        st.session_state["sample_after"]=s2
    before=after=None
    try:
        if before_file and after_file:
            before=read_image(before_file); after=read_image(after_file)
        elif "sample_before" in st.session_state and "sample_after" in st.session_state:
            before=cv2.cvtColor(np.array(st.session_state["sample_before"]),cv2.COLOR_RGB2BGR)
            after=cv2.cvtColor(np.array(st.session_state["sample_after"]),cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Could not read the uploaded image: {e}")
    if before is not None and after is not None:
        with st.spinner("Preprocessing images and detecting changes..."):
            b,a,mask,overlay,heat,pct,severity,regions,aligned=analyze(before,after,threshold,min_area)
        st.success("Image comparison completed.")
        if not aligned:
            st.warning("Automatic image alignment was not successful. Images were resized to the same dimensions; use images with similar viewpoints for better results.")
        m1,m2,m3,m4=st.columns(4)
        m1.markdown(f'<div class="metric"><small>Changed area</small><h2>{pct:.2f}%</h2></div>',unsafe_allow_html=True)
        m2.markdown(f'<div class="metric"><small>Severity</small><h2>{severity}</h2></div>',unsafe_allow_html=True)
        m3.markdown(f'<div class="metric"><small>Detected regions</small><h2>{len(regions)}</h2></div>',unsafe_allow_html=True)
        m4.markdown(f'<div class="metric"><small>Image size</small><h2>{b.shape[1]} × {b.shape[0]}</h2></div>',unsafe_allow_html=True)
        st.write("")
        left,right=st.columns(2)
        with left:
            st.markdown("**Before image**")
            st.image(rgb(b),use_container_width=True)
        with right:
            st.markdown("**After image**")
            st.image(rgb(a),use_container_width=True)
        st.subheader("Damage and change assessment")
        st.write(f"**Study area:** {location}  |  **Selected category:** {category}")
        if severity=="High":
            st.error("HIGH CHANGE ALERT: Significant visual differences detected. Prioritize field verification.")
        elif severity=="Medium":
            st.warning("MEDIUM CHANGE ALERT: Noticeable visual differences detected. Review affected regions.")
        else:
            st.info("LOW CHANGE: Limited visual differences detected. Continue routine monitoring.")
        chart=pd.DataFrame({"Class":["Changed","Unchanged"],"Area (%)":[pct,100-pct]})
        st.plotly_chart(px.pie(chart,names="Class",values="Area (%)",hole=.55,
                               title="Proportion of changed and unchanged pixels"),use_container_width=True)
        st.session_state["analysis"]={"before":b,"after":a,"mask":mask,"overlay":overlay,"heat":heat,
             "pct":pct,"severity":severity,"regions":regions,"category":category,"location":location,
             "aligned":aligned,"threshold":threshold,"min_area":min_area}
    else:
        st.info("Upload both images above, or click **Load sample images** to test the application without your own dataset.")
        st.markdown('<div class="note">For meaningful results, use pre-event and post-event images of the same geographical area with similar resolution, angle, and lighting.</div>',unsafe_allow_html=True)

with tab2:
    st.subheader("Change maps and affected-region visualization")
    if "analysis" in st.session_state:
        d=st.session_state["analysis"]
        v1,v2=st.columns(2)
        with v1:
            st.markdown("**Detected change mask**")
            st.image(d["mask"],caption="White pixels represent detected visual differences",use_container_width=True,clamp=True)
        with v2:
            st.markdown("**Change overlay**")
            st.image(rgb(d["overlay"]),caption="Red overlay highlights detected changes",use_container_width=True)
        st.markdown("**Difference heatmap**")
        st.image(rgb(d["heat"]),use_container_width=True)
        if d["regions"]:
            st.markdown("**Detected regions (image pixel coordinates)**")
            st.dataframe(pd.DataFrame(d["regions"]),use_container_width=True,hide_index=True)
        else:
            st.info("No regions exceeded the selected minimum region size.")
    else:
        st.info("Run an image comparison in the Dashboard tab to view change maps.")

with tab3:
    st.subheader("Automated impact report")
    if "analysis" in st.session_state:
        d=st.session_state["analysis"]
        st.write(f"**Location:** {d['location']}  \n**Monitoring category:** {d['category']}  \n**Analysis time:** {datetime.now().strftime('%d %b %Y, %I:%M %p')}  \n**Changed area:** {d['pct']:.2f}%  \n**Severity:** {d['severity']}  \n**Detected regions:** {len(d['regions'])}")
        rows=d["regions"] or [{"Region":"No qualifying regions","X":"","Y":"","Width":"","Height":"","Area (pixels)":0}]
        report_df=pd.DataFrame(rows)
        csv=report_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download region analysis (CSV)",csv,
                           file_name="geoch ange_regions.csv".replace(" ",""),mime="text/csv")
        summary=f"""SATELLITE / DRONE CHANGE DETECTION REPORT
Location: {d['location']}
Category: {d['category']}
Generated: {datetime.now().isoformat(timespec='minutes')}
Changed area: {d['pct']:.2f}%
Severity: {d['severity']}
Detected regions: {len(d['regions'])}

Method: image resizing, ECC-based alignment attempt, LAB color difference,
thresholding, morphological filtering and connected-component analysis.

Important: This is a visual-change screening prototype. Detected changes are
not automatically confirmed disaster damage. Verify results with ground data
and expert assessment before operational decisions.
"""
        st.download_button("⬇️ Download summary report (TXT)",summary.encode("utf-8"),
                           file_name="geoch ange_report.txt".replace(" ",""),mime="text/plain")
    else:
        st.info("Complete an image comparison first to generate a report.")
    st.divider()
    st.markdown("### Computer Vision workflow")
    st.markdown("""
    1. **Image collection:** Upload pre-event and post-event satellite/drone images.
    2. **Preprocessing:** Convert to a common size and attempt affine image alignment.
    3. **Feature comparison:** Compare pixel-level LAB color information.
    4. **Change detection:** Apply thresholding and morphological filtering.
    5. **Region analysis:** Extract connected regions and calculate changed-pixel percentage.
    6. **Severity screening:** Assign Low, Medium or High using configurable prototype thresholds.
    7. **Reporting:** Visualize the mask, overlay, heatmap and export region results.
    """)
    st.markdown('<div class="note"><b>Academic implementation note:</b> This version implements the core change-detection and severity-screening workflow using OpenCV. It does not claim to be a trained deep-learning model or a validated semantic classifier for floods, buildings, vegetation or landslides. The monitoring category is selected by the user; the detected regions represent visual differences that need verification.</div>',unsafe_allow_html=True)

st.markdown("---")
st.caption("GeoChange Capstone Prototype | Computer Vision | Satellite/Drone Image Change Detection for Climate & Disaster Monitoring")
