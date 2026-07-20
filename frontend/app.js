const API = "http://127.0.0.1:8000/api";
const login = document.querySelector("#login"), chat = document.querySelector("#chat");
const messages = document.querySelector("#messages"), prompt = document.querySelector("#prompt");
let session = JSON.parse(localStorage.getItem("smartdesk-session") || "null");
let threadId = session ? localStorage.getItem(`smartdesk-thread:${session.user_id}`) : null;

function showChat() { login.classList.add("hidden"); chat.classList.remove("hidden"); }
function bubble(role, text, meta="") { const el=document.createElement("div"); el.className=`message ${role}`; const match=text.match(/(?:^|\n)Reference:\s*([^\n]+)\s*$/i); if(match){text=text.slice(0,match.index).trimEnd();meta=`Reference: ${match[1].trim()}`;} el.textContent=text; if(meta){const m=document.createElement("div");m.className="meta";m.textContent=meta;el.append(m)} messages.append(el); messages.scrollTop=messages.scrollHeight; }
async function request(path, options={}) { const res=await fetch(API+path, options); const body=await res.json(); if(!res.ok) throw new Error(body.detail || "Request failed"); return body; }

if (session) { showChat(); if (threadId) { request(`/history/${threadId}?user_id=${session.user_id}`,{headers:{"X-User-Token":session.token}}).then(d=>d.messages.forEach(m=>bubble(m.role,m.content))).catch(()=>{}); } }
document.querySelector("#loginForm").addEventListener("submit", async e => {
  e.preventDefault(); const username=document.querySelector("#username").value.trim();
  const saved=JSON.parse(localStorage.getItem(`smartdesk:${username.toLowerCase()}`) || "null");
  try { const data=await request("/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username,token:saved?.token})}); session={username,user_id:data.user_id,token:data.token||saved.token}; localStorage.setItem(`smartdesk:${username.toLowerCase()}`,JSON.stringify(session)); localStorage.setItem("smartdesk-session",JSON.stringify(session)); threadId=localStorage.getItem(`smartdesk-thread:${session.user_id}`); showChat(); } catch(err) { alert(err.message); }
});
document.querySelector("#chatForm").addEventListener("submit", async e => {
  e.preventDefault(); const text=prompt.value.trim(); if(!text)return; bubble("user",text); prompt.value=""; chat.classList.add("busy");
  try { const data=await request("/chat",{method:"POST",headers:{"Content-Type":"application/json","X-User-Token":session.token},body:JSON.stringify({user_id:session.user_id,thread_id:threadId,message:text})}); threadId=data.thread_id; localStorage.setItem(`smartdesk-thread:${session.user_id}`,threadId); bubble("assistant",data.answer,data.active_agent||""); } catch(err) { bubble("assistant",`Error: ${err.message}`); } finally { chat.classList.remove("busy"); prompt.focus(); }
});
document.querySelector("#newChat").addEventListener("click",()=>{threadId=null;localStorage.removeItem(`smartdesk-thread:${session.user_id}`);messages.replaceChildren();});
