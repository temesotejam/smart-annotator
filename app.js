import { loadAI, encodeImage, segment, isModelLoaded, clearEncodedImage } from "./ai.js";

"use strict";

const fileInput=document.getElementById("fileInput");
const canvas=document.getElementById("canvas");
const ctx=canvas.getContext("2d");
const stageWrap=document.getElementById("stageWrap");
const emptyState=document.getElementById("emptyState");
const imageList=document.getElementById("imageList");
const classList=document.getElementById("classList");
const imageStatus=document.getElementById("imageStatus");
const annotationStatus=document.getElementById("annotationStatus");
const selectionPanel=document.getElementById("selectionPanel");
const aiStatus=document.getElementById("aiStatus");
const aiBadge=document.getElementById("aiBadge");
const aiLoadBtn=document.getElementById("aiLoadBtn");
const aiAcceptBtn=document.getElementById("aiAcceptBtn");
const aiClearBtn=document.getElementById("aiClearBtn");

const palette=["#5b8cff","#ff7a59","#32c48d","#f3c84b","#b47cff","#31c6d4","#ff5d92","#9bc53d"];
let images=[];
let currentIndex=-1;
let classes=[{id:1,name:"object",color:palette[0]}];
let activeClassId=1;
let tool="box";
let selectedId=null;
let drawing=null;
let history=[];
let aiPoints=[];
let aiCandidate=null;
let aiBusy=false;
let aiPreparedImageId=null;

function uid(){return crypto.randomUUID?crypto.randomUUID():String(Date.now())+Math.random()}
function current(){return images[currentIndex]||null}
function activeClass(){return classes.find(c=>c.id===activeClassId)||classes[0]}
function pushHistory(){
  history.push(JSON.stringify(images.map(i=>({id:i.id,annotations:i.annotations}))));
  if(history.length>30)history.shift();
}
function restoreHistory(){
  const s=history.pop(); if(!s)return;
  const snapshot=JSON.parse(s);
  for(const item of snapshot){const im=images.find(x=>x.id===item.id);if(im)im.annotations=item.annotations}
  selectedId=null; renderAll();
}
function resizeCanvas(){
  const r=stageWrap.getBoundingClientRect();
  const dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.floor(r.width*dpr));
  canvas.height=Math.max(1,Math.floor(r.height*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
new ResizeObserver(resizeCanvas).observe(stageWrap);

async function addFiles(files){
  const accepted=[...files].filter(f=>f.type.startsWith("image/"));
  for(const file of accepted){
    const url=URL.createObjectURL(file);
    const img=new Image();
    await new Promise((res,rej)=>{img.onload=res;img.onerror=rej;img.src=url});
    images.push({id:uid(),file,name:file.name,url,img,width:img.naturalWidth,height:img.naturalHeight,annotations:[]});
  }
  if(currentIndex<0&&images.length)currentIndex=0;
  renderAll();
}
fileInput.addEventListener("change",e=>addFiles(e.target.files));
stageWrap.addEventListener("dragover",e=>{e.preventDefault()});
stageWrap.addEventListener("drop",e=>{e.preventDefault();addFiles(e.dataTransfer.files)});

function viewTransform(){
  const im=current(); if(!im)return null;
  const w=stageWrap.clientWidth,h=stageWrap.clientHeight;
  const s=Math.min(w/im.width,h/im.height)*0.94;
  return {s,ox:(w-im.width*s)/2,oy:(h-im.height*s)/2};
}
function canvasToImage(x,y){
  const t=viewTransform(); if(!t)return null;
  return {x:(x-t.ox)/t.s,y:(y-t.oy)/t.s};
}
function imageToCanvas(p){
  const t=viewTransform();return{x:t.ox+p.x*t.s,y:t.oy+p.y*t.s}
}
function insideImage(p){
  const im=current();return im&&p.x>=0&&p.y>=0&&p.x<=im.width&&p.y<=im.height;
}

function draw(){
  const w=stageWrap.clientWidth,h=stageWrap.clientHeight;
  ctx.clearRect(0,0,w,h);
  const im=current();
  emptyState.style.display=im?"none":"flex";
  if(!im)return;
  const t=viewTransform();
  ctx.drawImage(im.img,t.ox,t.oy,im.width*t.s,im.height*t.s);
  for(const a of im.annotations)drawAnnotation(a,a.id===selectedId);
  if(drawing){
    if(drawing.type==="box")drawBox(drawing,true);
    else if(drawing.type==="polygon")drawPolygon(drawing,true);
  }
  if(aiCandidate?.polygon?.length){
    drawPolygon({type:"polygon",classId:activeClassId,points:aiCandidate.polygon,preview:false},true);
  }
  if(tool==="ai"&&aiPoints.length){
    for(const q of aiPoints){
      const p=imageToCanvas({x:q.x*im.width,y:q.y*im.height});
      ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);
      ctx.fillStyle=q.label===1?"#52e39a":"#ff6675";ctx.fill();
      ctx.lineWidth=2;ctx.strokeStyle="#ffffff";ctx.stroke();
      ctx.beginPath();ctx.moveTo(p.x-4,p.y);ctx.lineTo(p.x+4,p.y);
      if(q.label===1){ctx.moveTo(p.x,p.y-4);ctx.lineTo(p.x,p.y+4)}
      ctx.strokeStyle="#ffffff";ctx.lineWidth=1.5;ctx.stroke();
    }
  }
}
function drawAnnotation(a,selected){
  if(a.type==="box")drawBox(a,selected); else drawPolygon(a,selected);
}
function styleFor(a,selected){
  const c=classes.find(x=>x.id===a.classId)||activeClass();
  ctx.strokeStyle=c?.color||"#5b8cff";
  ctx.fillStyle=(c?.color||"#5b8cff")+"2b";
  ctx.lineWidth=selected?3:2;
}
function drawBox(a,selected=false){
  styleFor(a,selected);
  const p1=imageToCanvas({x:a.x,y:a.y});
  const p2=imageToCanvas({x:a.x+a.w,y:a.y+a.h});
  ctx.beginPath();ctx.rect(p1.x,p1.y,p2.x-p1.x,p2.y-p1.y);ctx.fill();ctx.stroke();
}
function drawPolygon(a,selected=false){
  if(!a.points?.length)return;
  styleFor(a,selected);
  ctx.beginPath();
  const p0=imageToCanvas(a.points[0]);ctx.moveTo(p0.x,p0.y);
  for(let i=1;i<a.points.length;i++){const p=imageToCanvas(a.points[i]);ctx.lineTo(p.x,p.y)}
  if(!a.preview)ctx.closePath();
  if(a.points.length>2&&!a.preview)ctx.fill();
  ctx.stroke();
  for(const q of a.points){const p=imageToCanvas(q);ctx.beginPath();ctx.arc(p.x,p.y,3.5,0,Math.PI*2);ctx.fillStyle=ctx.strokeStyle;ctx.fill()}
}
function eventPoint(e){
  const r=canvas.getBoundingClientRect();return canvasToImage(e.clientX-r.left,e.clientY-r.top)
}

canvas.addEventListener("pointerdown",e=>{
  const p=eventPoint(e); if(!insideImage(p))return;
  if(tool==="box"){
    pushHistory(); drawing={id:uid(),type:"box",classId:activeClassId,x:p.x,y:p.y,w:0,h:0}; canvas.setPointerCapture(e.pointerId);
  }else if(tool==="polygon"){
    if(!drawing){pushHistory();drawing={id:uid(),type:"polygon",classId:activeClassId,points:[p],preview:true}}
    else drawing.points.push(p);
    draw();
  }else if(tool==="select"){
    selectedId=hitTest(p); updateSelection(); draw();
  }else if(tool==="ai"){
    handleAIPoint(e,p);
  }
});
canvas.addEventListener("pointermove",e=>{
  if(!drawing||drawing.type!=="box")return;
  const p=eventPoint(e);
  drawing.w=p.x-drawing.x;drawing.h=p.y-drawing.y;draw();
});
canvas.addEventListener("pointerup",e=>{
  if(!drawing||drawing.type!=="box")return;
  let a=drawing;drawing=null;
  if(a.w<0){a.x+=a.w;a.w=-a.w} if(a.h<0){a.y+=a.h;a.h=-a.h}
  if(a.w>2&&a.h>2){current().annotations.push(a);selectedId=a.id}
  renderAll();
});
canvas.addEventListener("contextmenu",e=>{if(tool==="ai")e.preventDefault()});

window.addEventListener("keydown",e=>{
  if(e.key==="Enter"&&tool==="ai"&&aiCandidate){
    e.preventDefault();acceptAICandidate();
  }else if(e.key==="Escape"&&tool==="ai"&&(aiCandidate||aiPoints.length)){
    e.preventDefault();clearAICandidate();
  }else if(e.key==="Enter"&&drawing?.type==="polygon"){
    if(drawing.points.length>=3){drawing.preview=false;current().annotations.push(drawing);selectedId=drawing.id}
    drawing=null;renderAll();
  }else if(e.key==="Escape"&&drawing){drawing=null;renderAll()}
  else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){e.preventDefault();restoreHistory()}
  else if(e.key==="Delete"||e.key==="Backspace"){if(selectedId)deleteSelected()}
});

function hitTest(p){
  const anns=[...current().annotations].reverse();
  for(const a of anns){
    if(a.type==="box"&&p.x>=a.x&&p.x<=a.x+a.w&&p.y>=a.y&&p.y<=a.y+a.h)return a.id;
    if(a.type==="polygon"&&pointInPolygon(p,a.points))return a.id;
  } return null;
}
function pointInPolygon(p,vs){
  let inside=false;
  for(let i=0,j=vs.length-1;i<vs.length;j=i++){
    const xi=vs[i].x,yi=vs[i].y,xj=vs[j].x,yj=vs[j].y;
    const intersect=((yi>p.y)!=(yj>p.y))&&(p.x<(xj-xi)*(p.y-yi)/(yj-yi+1e-12)+xi);
    if(intersect)inside=!inside;
  } return inside;
}
function deleteSelected(){
  const im=current();if(!im||!selectedId)return;
  pushHistory();im.annotations=im.annotations.filter(a=>a.id!==selectedId);selectedId=null;renderAll();
}
document.getElementById("deleteBtn").onclick=deleteSelected;
document.getElementById("undoBtn").onclick=restoreHistory;

document.querySelectorAll(".tool").forEach(b=>b.onclick=async()=>{
  tool=b.dataset.tool;drawing=null;
  document.querySelectorAll(".tool").forEach(x=>x.classList.toggle("active",x===b));
  canvas.style.cursor=tool==="select"?"default":"crosshair";
  if(tool!=="ai") clearAICandidate(false);
  draw();
  if(tool==="ai"&&current()) await prepareAIForCurrent();
});

document.getElementById("addClassBtn").onclick=()=>{
  const name=prompt("クラス名を入力してください");if(!name?.trim())return;
  const next=Math.max(0,...classes.map(c=>c.id))+1;
  classes.push({id:next,name:name.trim(),color:palette[(next-1)%palette.length]});activeClassId=next;renderClasses();draw();
};

function renderClasses(){
  classList.innerHTML="";
  for(const c of classes){
    const el=document.createElement("div");el.className="class-item"+(c.id===activeClassId?" active":"");
    el.innerHTML='<span class="swatch" style="background:'+c.color+'"></span><span class="class-name"></span>';
    el.querySelector(".class-name").textContent=c.name;
    el.onclick=()=>{activeClassId=c.id;renderClasses()};
    classList.appendChild(el);
  }
}
function renderImages(){
  if(!images.length){imageList.className="image-list empty";imageList.textContent="画像を追加してください";return}
  imageList.className="image-list";imageList.innerHTML="";
  images.forEach((im,i)=>{
    const el=document.createElement("div");el.className="image-item"+(i===currentIndex?" active":"");
    el.innerHTML='<img class="thumb"><div class="image-meta"><div class="image-name"></div><div class="image-count"></div></div>';
    el.querySelector("img").src=im.url;el.querySelector(".image-name").textContent=im.name;
    el.querySelector(".image-count").textContent=im.annotations.length+" annotations";
    el.onclick=()=>{currentIndex=i;selectedId=null;drawing=null;clearAICandidate(false);aiPreparedImageId=null;clearEncodedImage();renderAll();if(tool==="ai")prepareAIForCurrent()};
    imageList.appendChild(el);
  });
}
function updateSelection(){
  const a=current()?.annotations.find(x=>x.id===selectedId);
  if(!a){selectionPanel.className="selection-panel muted";selectionPanel.textContent="アノテーションを選択してください";return}
  const c=classes.find(x=>x.id===a.classId);
  selectionPanel.className="selection-panel";
  selectionPanel.innerHTML="<b>Class:</b> "+escapeHtml(c?.name||"?")+"<br><b>Type:</b> "+a.type;
}
function renderAll(){
  renderImages();renderClasses();updateSelection();
  imageStatus.textContent=images.length?(currentIndex+1)+" / "+images.length:"0 / 0";
  annotationStatus.textContent=(current()?.annotations.length||0)+" annotations";
  draw();
}
function escapeHtml(s){return s.replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]))}

function download(name,data,type="application/json"){
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([data],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
document.getElementById("exportProjectBtn").onclick=()=>{
  const data={version:1,classes,images:images.map(i=>({name:i.name,width:i.width,height:i.height,annotations:i.annotations}))};
  download("smart-annotator-project.json",JSON.stringify(data,null,2));
};
document.getElementById("exportCocoBtn").onclick=()=>{
  const categories=classes.map(c=>({id:c.id,name:c.name,supercategory:""}));
  const cocoImages=images.map((im,i)=>({id:i+1,file_name:im.name,width:im.width,height:im.height}));
  const annotations=[];let aid=1;
  images.forEach((im,i)=>im.annotations.forEach(a=>{
    let segmentation=[],bbox,area;
    if(a.type==="box"){bbox=[a.x,a.y,a.w,a.h];area=a.w*a.h;segmentation=[[a.x,a.y,a.x+a.w,a.y,a.x+a.w,a.y+a.h,a.x,a.y+a.h]]}
    else{
      const xs=a.points.map(p=>p.x),ys=a.points.map(p=>p.y);const minx=Math.min(...xs),miny=Math.min(...ys),maxx=Math.max(...xs),maxy=Math.max(...ys);
      bbox=[minx,miny,maxx-minx,maxy-miny];segmentation=[a.points.flatMap(p=>[p.x,p.y])];area=polygonArea(a.points);
    }
    annotations.push({id:aid++,image_id:i+1,category_id:a.classId,segmentation,bbox,area,iscrowd:0});
  }));
  download("annotations.coco.json",JSON.stringify({info:{description:"Smart Annotator export"},images:cocoImages,annotations,categories},null,2));
};
function polygonArea(points){let a=0;for(let i=0,j=points.length-1;i<points.length;j=i++)a+=(points[j].x+points[i].x)*(points[j].y-points[i].y);return Math.abs(a/2)}
document.getElementById("exportYoloBtn").onclick=async()=>{
  if(!window.JSZip){alert("YOLO ZIP用ライブラリを読み込めませんでした");return}
  const zip=new JSZip();zip.file("classes.txt",classes.map(c=>c.name).join("\n"));
  images.forEach(im=>{
    const rows=[];
    im.annotations.forEach(a=>{
      const cls=classes.findIndex(c=>c.id===a.classId);if(cls<0)return;
      if(a.type==="box"){
        const cx=(a.x+a.w/2)/im.width,cy=(a.y+a.h/2)/im.height,w=a.w/im.width,h=a.h/im.height;
        rows.push([cls,cx,cy,w,h].map((v,i)=>i?v.toFixed(6):v).join(" "));
      }else{
        const vals=[cls,...a.points.flatMap(p=>[p.x/im.width,p.y/im.height])];
        rows.push(vals.map((v,i)=>i?v.toFixed(6):v).join(" "));
      }
    });
    zip.file(im.name.replace(/\.[^.]+$/,"")+".txt",rows.join("\n"));
  });
  const blob=await zip.generateAsync({type:"blob"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="yolo-labels.zip";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};

async function setAIState(kind,message){
  aiBadge.className="ai-badge"+(kind?" "+kind:"");
  aiBadge.textContent=kind==="ready"?"準備完了":kind==="busy"?"処理中":kind==="error"?"エラー":"未読込";
  if(message)aiStatus.textContent=message;
}

async function ensureAIModel(){
  if(isModelLoaded()){setAIState("ready","AIモデル準備完了");return true}
  try{
    aiBusy=true;
    setAIState("busy","AIモデルを読み込んでいます…");
    aiLoadBtn.disabled=true;
    await loadAI(msg=>setAIState("busy",msg));
    setAIState("ready","AIモデル準備完了。画像を解析します。");
    return true;
  }catch(err){
    console.error(err);
    setAIState("error",err.message||"AIモデルの読み込みに失敗しました。");
    return false;
  }finally{
    aiBusy=false;
    aiLoadBtn.disabled=false;
  }
}

async function prepareAIForCurrent(){
  const im=current();
  if(!im)return false;
  if(aiBusy)return false;
  if(aiPreparedImageId===im.id){
    setAIState("ready","対象を左クリックしてください。右クリックで除外点を追加できます。");
    return true;
  }
  const ok=await ensureAIModel();
  if(!ok)return false;
  try{
    aiBusy=true;
    setAIState("busy","この画像をAI用に解析しています…");
    await encodeImage(im.id,im.url,msg=>setAIState("busy",msg));
    aiPreparedImageId=im.id;
    setAIState("ready","対象を左クリックしてください。右クリックで除外点を追加できます。");
    return true;
  }catch(err){
    console.error(err);
    setAIState("error",err.message||"画像のAI解析に失敗しました。");
    return false;
  }finally{
    aiBusy=false;
  }
}

async function handleAIPoint(e,p){
  if(aiBusy)return;
  const im=current();
  if(!im)return;
  if(e.button!==0&&e.button!==2)return;

  const ready=await prepareAIForCurrent();
  if(!ready)return;

  aiPoints.push({
    x:p.x/im.width,
    y:p.y/im.height,
    label:e.button===2?0:1
  });
  aiClearBtn.disabled=false;
  draw();

  try{
    aiBusy=true;
    setAIState("busy","輪郭を計算しています…");
    const result=await segment(aiPoints);
    aiCandidate=result;
    aiAcceptBtn.disabled=false;
    setAIState("ready","候補生成完了（score "+result.score.toFixed(2)+"）。Enterまたは「AI確定」で採用できます。");
    draw();
  }catch(err){
    console.error(err);
    setAIState("error",err.message||"輪郭生成に失敗しました。");
  }finally{
    aiBusy=false;
  }
}

function acceptAICandidate(){
  if(!aiCandidate?.polygon?.length||!current())return;
  pushHistory();
  const a={
    id:uid(),
    type:"polygon",
    classId:activeClassId,
    points:aiCandidate.polygon.map(p=>({x:p.x,y:p.y})),
    preview:false,
    source:"ai"
  };
  current().annotations.push(a);
  selectedId=a.id;
  clearAICandidate(false);
  renderAll();
  setAIState("ready","AI輪郭を追加しました。次の対象をクリックできます。");
}

function clearAICandidate(redraw=true){
  aiPoints=[];
  aiCandidate=null;
  aiAcceptBtn.disabled=true;
  aiClearBtn.disabled=true;
  if(redraw)draw();
}

aiLoadBtn.onclick=async()=>{
  await ensureAIModel();
  if(current())await prepareAIForCurrent();
};
aiAcceptBtn.onclick=acceptAICandidate;
aiClearBtn.onclick=()=>{
  clearAICandidate();
  setAIState(isModelLoaded()?"ready":"","候補をクリアしました。");
};

renderAll();
