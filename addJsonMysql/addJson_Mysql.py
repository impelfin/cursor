import json
import pymysql

conn = pymysql.connect(
    host='192.168.1.43',
    user='mysql',
    password='1234',
    db='testdb',
    charset='utf8mb4'
)
cur = conn.cursor()

with open('lines.json', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        개방서비스명 = obj.get("개방서비스명")
        인허가일자 = obj.get("인허가일자")
        폐업일자 = obj.get("폐업일자")
        소재지전체주소 = obj.get("소재지전체주소")
        cur.execute(
            "INSERT INTO services (개방서비스명, 인허가일자, 폐업일자, 소재지전체주소) VALUES (%s, %s, %s, %s)",
            (개방서비스명, 인허가일자, 폐업일자, 소재지전체주소)
        )
conn.commit()
cur.close()
conn.close()
print("finish")