"""Read-only dashboard bound to loopback; no inference requests are sent."""
import argparse
import json
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Viper · Theo dõi API</title>
<style>body{font:16px system-ui;background:#f4f6fa;color:#182233;margin:30px auto;max-width:1250px;padding:0 20px}h1{font-size:26px}button,input,select{font:inherit;padding:8px;border:1px solid #ccd4df;border-radius:8px}button{cursor:pointer}#stats,#active{padding:16px;background:white;border-radius:12px;margin:16px 0}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #e3e7ee;vertical-align:top}th{position:sticky;top:0;background:#edf2f8}details{max-width:730px}summary{cursor:pointer;white-space:pre-wrap}p{white-space:pre-wrap}.ok{color:#137344}.bad{color:#b62929}.muted{color:#607086}#error{color:#b62929}</style>
<h1>Viper · Theo dõi câu hỏi và thời gian</h1><p class="muted">Trang riêng trên máy này · Tự cập nhật mỗi 2 giây · Không gửi thêm câu hỏi đến model.</p>
<select id="mode"><option value="live">API trực tiếp</option><option value="dev">Kết quả full dev</option></select> <input id="search" placeholder="Tìm ID hoặc câu hỏi" size="35"> <label><input id="only" type="checkbox"> Chỉ lỗi / câu sai</label>
<div id="error"></div><div id="stats"></div><div id="active"></div>
<table><thead><tr><th>ID / trạng thái</th><th>Câu hỏi, dự đoán, giải thích</th><th>Thời gian</th></tr></thead><tbody id="rows"></tbody></table>
<script>
const $=id=>document.getElementById(id);let data=null;
function el(tag,text,cls){let e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e}
function render(){if(!data)return;let dev=$('mode').value==='dev',list=dev?data.dev:data.live;
let good=list.filter(r=>dev?r.score?.tolerant&&r.status==='ok':r.status==='ok').length;
let times=list.map(r=>r.seconds).filter(v=>typeof v==='number');
$('stats').textContent=`${dev?'Dev':'Log API'}: ${list.length} câu · ${dev?'Đúng':'Thành công'}: ${good} · ${dev?'Sai':'Lỗi'}: ${list.length-good} · Trung bình: ${(times.reduce((a,b)=>a+b,0)/(times.length||1)).toFixed(2)} giây`;
$('active').replaceChildren();if(dev){$('active').textContent='Full dev đã hoàn tất. Thời gian ở đây gồm lượt gọi HTTP công khai.'}else if(!data.active.length){$('active').textContent='Không có câu đang xử lý được ghi nhận. Thời gian dưới đây đo ở server.'}else{for(let r of data.active){$('active').append(el('p',`${r.query_id} · Đang chờ / xử lý ${(Date.now()/1000-r.time_unix).toFixed(1)} giây\n${r.query}`))}}
let search=$('search').value.toLowerCase();$('rows').replaceChildren();
for(let r of [...list].reverse().filter(r=>(`${r.query_id} ${r.query||r.question||''}`).toLowerCase().includes(search)&&(!$('only').checked||(dev?!r.score?.tolerant||r.status!=='ok':r.status!=='ok')))){
let tr=el('tr',''),id=el('td','');id.append(el('b',r.query_id),el('div',dev?(r.score?.tolerant&&r.status==='ok'?'Đúng':'Sai'):r.status,dev?(r.score?.tolerant&&r.status==='ok'?'ok':'bad'):(r.status==='ok'?'ok':'bad')));
let cell=el('td',''),d=el('details','');d.append(el('summary',r.query||r.question||''));let p=r.prediction||{};d.append(el('p',`Dự đoán: ${p.answer??'—'} ${p.unit??''}\n${dev?`Đáp án dev: ${r.gold_answer} ${r.gold_unit}\n`:''}Giải thích: ${p.explanation??'—'}${r.error?'\nLỗi: '+r.error:''}`));cell.append(d);tr.append(id,cell,el('td',typeof r.seconds==='number'?r.seconds.toFixed(2)+' s':'—'));$('rows').append(tr)}
}
async function refresh(){try{let r=await fetch('/data',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);data=await r.json();$('error').textContent='';render()}catch(e){$('error').textContent='Không đọc được monitor: '+e.message}}
for(let id of ['mode','search','only'])$(id).addEventListener('input',render);refresh();setInterval(refresh,2000);
</script></html>'''

def read_jsonl(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A writer may still be appending the last line.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                content, mime = PAGE.encode(), "text/html; charset=utf-8"
            elif self.path == "/data":
                pending = {}
                for row in read_jsonl(args.logs / "activity.jsonl"):
                    key = (row.get("query_id"), row.get("time_unix"))
                    if row.get("status") == "processing":
                        pending[key] = row
                    else:
                        pending.pop(key, None)
                # An interrupted process can leave a start event with no finish.
                active = [r for r in pending.values() if time.time()-r["time_unix"] < 65]
                content = json.dumps({"live": read_jsonl(args.logs / "requests.jsonl"),
                                      "active": active, "dev": read_jsonl(args.dev)},
                                     ensure_ascii=False).encode()
                mime = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
