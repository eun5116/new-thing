# app.py - 고급 연예인 이미지 검색기 v3.0
# 실행: streamlit run app.py

import os, io, json, time, random, re, hashlib, base64
from pathlib import Path
from urllib.parse import urlparse, quote_plus, urljoin
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import requests
from PIL import Image, ImageFilter
import imagehash
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Config
load_dotenv()
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
TIMEOUT = 15.0

# Streamlit 페이지 설정
st.set_page_config(page_title="연예인 이미지 파인더 v3.0", page_icon="🌟", layout="wide")

# 얼굴 인식 라이브러리 체크 및 초기화
FACE_LIBS = {"face_recognition": None, "opencv": None, "mediapipe": None}

try:
    import face_recognition
    FACE_LIBS["face_recognition"] = face_recognition
    print("✅ face_recognition 로드됨")
except ImportError:
    print("❌ face_recognition 미설치")

try:
    import cv2
    FACE_LIBS["opencv"] = cv2
    print("✅ OpenCV 로드됨")
except ImportError:
    print("❌ OpenCV 미설치")

try:
    import mediapipe as mp
    FACE_LIBS["mediapipe"] = mp
    print("✅ MediaPipe 로드됨")
except ImportError:
    print("❌ MediaPipe 미설치")

# DuckDuckGo 검색
try:
    from duckduckgo_search import DDGS
    HAVE_DDG = True
except ImportError:
    HAVE_DDG = False

# Session state 초기화
def _init_state():
    st.session_state.setdefault("results", [])
    st.session_state.setdefault("query_name", "")
    st.session_state.setdefault("reference_faces", [])
    st.session_state.setdefault("options", {})
    st.session_state.setdefault("notice", "")
    st.session_state.setdefault("last_search_ok", False)
_init_state()

# 메모리 저장
MEM_DIR = Path.home() / ".celeb_finder_v3"
MEM_DIR.mkdir(parents=True, exist_ok=True)
MEM_PATH = MEM_DIR / "memory.json"

def load_memory() -> Dict[str, Any]:
    if MEM_PATH.exists():
        try:
            return json.loads(MEM_PATH.read_text(encoding="utf-8"))
        except:
            pass
    return {
        "blocked_urls": [],
        "blocked_hosts": [],
        "blocked_hashes": [],
        "preferred_hosts": [],
        "celebrity_faces": {},  # 연예인별 다중 얼굴 데이터
        "search_history": {},
        "version": 0,
    }

def save_memory(mem: Dict[str, Any]):
    mem["version"] = int(mem.get("version", 0)) + 1
    MEM_PATH.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")

MEM = load_memory()

# Data models
@dataclass
class FaceData:
    encoding: np.ndarray
    confidence: float
    source_url: str
    bbox: Tuple[int, int, int, int]  # (top, right, bottom, left)

@dataclass
class Candidate:
    url: str
    source: str
    thumb: Optional[str]
    width: Optional[int]
    height: Optional[int]
    meta: Dict[str, Any]
    search_engine: str = "unknown"

@dataclass
class ValidImage:
    url: str
    source: str
    width: int
    height: int
    size_bytes: int
    fmt: str
    phash: str
    focus_var: float
    score: float
    face_similarity: float = 0.0
    face_confidence: float = 0.0
    face_count: int = 0
    thumb: Optional[str] = None
    search_engine: str = "unknown"
    face_bbox: Optional[Tuple[int, int, int, int]] = None

# Session 초기화
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
})

# 유틸리티 함수들
def host_of(u: str) -> str:
    try:
        return urlparse(u).hostname or ""
    except:
        return ""

def focus_variance(img: Image.Image) -> float:
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)
    return float(np.var(gx) + np.var(gy))

def domain_weight(domain: str) -> float:
    if not domain: 
        return 0.0
    
    # 차단된 도메인
    if domain in MEM.get("blocked_hosts", []): 
        return -1.0
    
    # 선호 도메인
    if domain in MEM.get("preferred_hosts", []): 
        return 0.4
    
    # 고품질 소스 가중치
    high_quality = {
        "gettyimages": 0.5, "shutterstock": 0.4, "alamy": 0.3,
        "reuters": 0.4, "ap": 0.4, "afp": 0.4, "epa": 0.3,
        "dispatch.co.kr": 0.4, "newsen.com": 0.3, "osen.co.kr": 0.3,
        "sbs.co.kr": 0.3, "kbs.co.kr": 0.3, "mbc.co.kr": 0.3, "jtbc.co.kr": 0.3,
        "naver.com": 0.2, "daum.net": 0.2, "joins.com": 0.2,
        "instagram.com": 0.2, "twitter.com": 0.1, "facebook.com": 0.1,
        "soompi.com": 0.2, "allkpop.com": 0.2, "koreastardaily.com": 0.2
    }
    
    for source, weight in high_quality.items():
        if source in domain:
            return weight
    
    return 0.0

def normalize_instagram_handle(handle: str) -> str:
    handle = (handle or "").strip()
    if not handle:
        return ""
    if handle.startswith("http://") or handle.startswith("https://"):
        try:
            parsed = urlparse(handle)
            path = parsed.path.strip("/")
            handle = path.split("/")[0] if path else ""
        except:
            return ""
    handle = handle.lstrip("@").strip()
    return handle

def search_instagram_profile(handle: str) -> List[Candidate]:
    """Instagram 공개 프로필의 프로필 이미지 추출 (로그인 없이 가능한 범위)"""
    handle = normalize_instagram_handle(handle)
    if not handle:
        return []
    profile_url = f"https://www.instagram.com/{handle}/"
    try:
        headers = SESSION.headers.copy()
        headers.update({"Referer": "https://www.instagram.com/"})
        response = SESSION.get(profile_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        og_image = soup.find("meta", property="og:image")
        image_url = og_image.get("content") if og_image else ""
        if image_url and image_url.startswith("http"):
            return [Candidate(
                url=image_url,
                source="instagram.com",
                thumb=None,
                width=None,
                height=None,
                meta={"from": "instagram_profile", "profile": handle},
                search_engine="instagram"
            )]
    except Exception as e:
        print(f"Instagram 프로필 검색 오류: {e}")
    return []

# 향상된 얼굴 인식 함수들
def extract_face_encodings_advanced(image_bytes: bytes) -> List[FaceData]:
    """고급 얼굴 인식 - 다중 라이브러리 사용"""
    face_data_list = []
    
    # face_recognition 사용 (가장 정확)
    if FACE_LIBS["face_recognition"]:
        try:
            face_recognition = FACE_LIBS["face_recognition"]
            
            # BytesIO를 통해 이미지 로드
            img = face_recognition.load_image_from_file(io.BytesIO(image_bytes))
            
            # 얼굴 위치 찾기
            face_locations = face_recognition.face_locations(img, model="hog")  # CNN이 더 정확하지만 느림
            
            if face_locations:
                # 얼굴 인코딩 생성
                face_encodings = face_recognition.face_encodings(img, face_locations)
                
                for encoding, location in zip(face_encodings, face_locations):
                    # 얼굴 크기로 신뢰도 계산
                    top, right, bottom, left = location
                    face_area = (right - left) * (bottom - top)
                    img_area = img.shape[0] * img.shape[1]
                    confidence = min(face_area / (img_area * 0.05), 1.0)  # 얼굴이 이미지의 5% 이상이면 높은 신뢰도
                    
                    face_data_list.append(FaceData(
                        encoding=encoding,
                        confidence=confidence,
                        source_url="",
                        bbox=location
                    ))
                    
        except Exception as e:
            print(f"face_recognition 오류: {e}")
    
    # MediaPipe 보완 사용
    if not face_data_list and FACE_LIBS["mediapipe"]:
        try:
            mp = FACE_LIBS["mediapipe"]
            mp_face_detection = mp.solutions.face_detection
            mp_drawing = mp.solutions.drawing_utils
            
            # PIL Image로 변환
            pil_img = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(pil_img)
            
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(img_array)
                
                if results.detections:
                    for detection in results.detections:
                        # MediaPipe는 인코딩을 제공하지 않으므로 간단한 특징 벡터 생성
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = img_array.shape
                        
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # 얼굴 영역 추출
                        face_img = img_array[y:y+height, x:x+width]
                        
                        # 간단한 히스토그램 기반 인코딩
                        if face_img.size > 0:
                            if FACE_LIBS["opencv"]:
                                cv2 = FACE_LIBS["opencv"]
                                face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY) if len(face_img.shape) == 3 else face_img
                                hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
                                encoding = hist.flatten()
                            else:
                                face_gray = np.mean(face_img, axis=2).astype(np.uint8) if len(face_img.shape) == 3 else face_img
                                hist, _ = np.histogram(face_gray, bins=256, range=(0, 256))
                                encoding = hist.astype(np.float32)
                            
                            face_data_list.append(FaceData(
                                encoding=encoding,
                                confidence=detection.score[0],
                                source_url="",
                                bbox=(y, x+width, y+height, x)
                            ))
                            
        except Exception as e:
            print(f"MediaPipe 오류: {e}")
    
    return face_data_list

def compare_faces_advanced(encoding1: np.ndarray, encoding2: np.ndarray, method: str = "face_recognition") -> float:
    """고급 얼굴 비교"""
    try:
        if method == "face_recognition" and FACE_LIBS["face_recognition"]:
            face_recognition = FACE_LIBS["face_recognition"]
            distance = face_recognition.face_distance([encoding1], encoding2)[0]
            return max(0, 1 - distance)
        else:
            # 코사인 유사도
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return max(0, min(1, dot_product / (norm1 * norm2)))
    except:
        return 0.0

def collect_reference_faces(celebrity_name: str, max_samples: int = 5) -> List[FaceData]:
    """연예인의 기준 얼굴들을 여러 소스에서 수집"""
    
    # 메모리에서 기존 데이터 확인
    if celebrity_name in MEM.get("celebrity_faces", {}):
        try:
            stored_faces = MEM["celebrity_faces"][celebrity_name]
            face_data_list = []
            for face_info in stored_faces:
                encoding = np.array(face_info["encoding"])
                face_data_list.append(FaceData(
                    encoding=encoding,
                    confidence=face_info["confidence"],
                    source_url=face_info["source_url"],
                    bbox=tuple(face_info["bbox"])
                ))
            if len(face_data_list) >= 3:  # 최소 3개 이상 있으면 사용
                return face_data_list
        except:
            pass
    
    st.info(f"🔍 {celebrity_name}의 기준 얼굴을 수집하는 중...")
    
    face_data_list = []
    search_queries = [
        f"{celebrity_name} 프로필 사진 고화질",
        f"{celebrity_name} 포토 공식",
        f"{celebrity_name} 얼굴 클로즈업",
        f"{celebrity_name} headshot portrait"
    ]
    
    for query in search_queries:
        if len(face_data_list) >= max_samples:
            break
            
        try:
            # Google에서 고품질 이미지 검색
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch&tbs=isz:l"
            response = SESSION.get(search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                img_tags = soup.find_all('img')
                
                for img_tag in img_tags[:5]:
                    if len(face_data_list) >= max_samples:
                        break
                        
                    src = img_tag.get('src')
                    if src and src.startswith('http') and len(src) > 60:
                        try:
                            img_response = SESSION.get(src, timeout=8)
                            if img_response.status_code == 200:
                                face_data = extract_face_encodings_advanced(img_response.content)
                                
                                for face in face_data:
                                    if face.confidence > 0.3:  # 충분한 신뢰도
                                        face.source_url = src
                                        face_data_list.append(face)
                                        if len(face_data_list) >= max_samples:
                                            break
                        except:
                            continue
        except Exception as e:
            print(f"기준 얼굴 수집 오류: {e}")
            continue
        
        time.sleep(0.5)  # API 제한 방지
    
    # 메모리에 저장
    if face_data_list:
        if "celebrity_faces" not in MEM:
            MEM["celebrity_faces"] = {}
            
        MEM["celebrity_faces"][celebrity_name] = [
            {
                "encoding": face.encoding.tolist(),
                "confidence": face.confidence,
                "source_url": face.source_url,
                "bbox": face.bbox
            }
            for face in face_data_list
        ]
        save_memory(MEM)
        
        st.success(f"✅ {len(face_data_list)}개의 기준 얼굴을 수집했습니다!")
    else:
        st.warning("⚠️ 기준 얼굴을 찾지 못했습니다. 얼굴 인식 없이 진행합니다.")
    
    return face_data_list

# 확장된 검색 소스들
def search_google_images_advanced(query: str, max_results: int) -> List[Candidate]:
    """향상된 Google 이미지 검색"""
    candidates = []
    
    # 다양한 Google 검색 옵션
    search_variations = [
        f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch&tbs=isz:l",  # 큰 이미지
        f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch&tbs=itp:photo",  # 사진만
        f"https://www.google.com/search?q={quote_plus(query + ' HD')}&tbm=isch",  # HD 이미지
    ]
    
    for search_url in search_variations[:2]:  # 2개 변형만 사용
        try:
            response = SESSION.get(search_url, timeout=12)
            response.raise_for_status()
            
            # JSON 데이터에서 이미지 URL 추출 (더 정확한 방법)
            text = response.text
            
            # Google 이미지의 JSON 응답에서 이미지 URL 패턴 찾기
            patterns = [
                r'"ou":"([^"]+)"',  # Original URL
                r'"murl":"([^"]+)"',  # Medium URL
                r'https://[^"\s]*\.(?:jpg|jpeg|png|webp)[^"\s]*'
            ]
            
            for pattern in patterns:
                urls = re.findall(pattern, text)
                for url in urls:
                    if len(url) > 100 and len(candidates) < max_results:
                        # URL 디코딩
                        try:
                            clean_url = url.replace('\\u003d', '=').replace('\\/', '/')
                            candidates.append(Candidate(
                                url=clean_url,
                                source=host_of(clean_url),
                                thumb=None,
                                width=None,
                                height=None,
                                meta={"from": "google_advanced"},
                                search_engine="google"
                            ))
                        except:
                            continue
                            
        except Exception as e:
            print(f"Google 고급 검색 오류: {e}")
            continue
    
    return candidates[:max_results]

def search_yandex_images(query: str, max_results: int) -> List[Candidate]:
    """Yandex 이미지 검색 (러시아 검색엔진 - 다른 결과)"""
    candidates = []
    try:
        search_url = f"https://yandex.com/images/search?text={quote_plus(query)}&isize=large"
        headers = SESSION.headers.copy()
        headers.update({
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            "Referer": "https://yandex.com/"
        })
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Yandex JSON 응답에서 이미지 추출
            text = response.text
            url_pattern = r'"orig":"([^"]+)"'
            urls = re.findall(url_pattern, text)
            
            for url in urls[:max_results]:
                if len(url) > 50:
                    clean_url = url.replace('\\/', '/')
                    candidates.append(Candidate(
                        url=clean_url,
                        source=host_of(clean_url),
                        thumb=None,
                        width=None,
                        height=None,
                        meta={"from": "yandex"},
                        search_engine="yandex"
                    ))
                    
    except Exception as e:
        print(f"Yandex 검색 오류: {e}")
        
    return candidates

def search_baidu_images(query: str, max_results: int) -> List[Candidate]:
    """Baidu 이미지 검색 (중국 검색엔진)"""
    candidates = []
    try:
        search_url = f"https://image.baidu.com/search/index?tn=baiduimage&word={quote_plus(query)}&pn=0"
        headers = SESSION.headers.copy()
        headers.update({
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.baidu.com/"
        })
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            text = response.text
            # Baidu JSON에서 이미지 URL 추출
            url_pattern = r'"objURL":"([^"]+)"'
            urls = re.findall(url_pattern, text)
            
            for url in urls[:max_results]:
                if len(url) > 50:
                    candidates.append(Candidate(
                        url=url,
                        source=host_of(url),
                        thumb=None,
                        width=None,
                        height=None,
                        meta={"from": "baidu"},
                        search_engine="baidu"
                    ))
                    
    except Exception as e:
        print(f"Baidu 검색 오류: {e}")
        
    return candidates

def search_pinterest_images(query: str, max_results: int) -> List[Candidate]:
    """Pinterest 이미지 검색"""
    candidates = []
    try:
        search_url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}"
        headers = SESSION.headers.copy()
        headers.update({
            "X-Requested-With": "XMLHttpRequest",
        })
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            text = response.text
            # Pinterest JSON에서 이미지 URL 추출
            patterns = [
                r'"url": "([^"]*\.jpg[^"]*)"',
                r'"url": "([^"]*\.jpeg[^"]*)"',
                r'"url": "([^"]*\.png[^"]*)"'
            ]
            
            for pattern in patterns:
                urls = re.findall(pattern, text, re.IGNORECASE)
                for url in urls:
                    if len(candidates) < max_results and len(url) > 50:
                        clean_url = url.replace('\\/', '/').replace('\\u003d', '=')
                        candidates.append(Candidate(
                            url=clean_url,
                            source=host_of(clean_url),
                            thumb=None,
                            width=None,
                            height=None,
                            meta={"from": "pinterest"},
                            search_engine="pinterest"
                        ))
                        
    except Exception as e:
        print(f"Pinterest 검색 오류: {e}")
        
    return candidates[:max_results]

def search_korean_news_sites(query: str, max_results: int) -> List[Candidate]:
    """한국 뉴스 사이트에서 검색"""
    candidates = []
    
    news_sites = [
        "https://search.naver.com/search.naver?where=image&query=",
        "https://search.daum.net/search?w=img&q=",
        "https://www.google.com/search?q=site:dispatch.co.kr+OR+site:osen.co.kr+OR+site:newsen.com+",
    ]
    
    for base_url in news_sites[:2]:  # 처음 2개만 사용
        try:
            search_url = base_url + quote_plus(query)
            response = SESSION.get(search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 다양한 이미지 태그 패턴 찾기
                img_selectors = [
                    'img[data-source]',  # Naver
                    'img[src*="jpg"]',
                    'img[src*="jpeg"]',
                    'img[src*="png"]',
                    'img[data-original-src]'  # Daum
                ]
                
                for selector in img_selectors:
                    imgs = soup.select(selector)
                    for img in imgs:
                        if len(candidates) >= max_results:
                            break
                            
                        img_url = img.get('data-source') or img.get('src') or img.get('data-original-src')
                        if img_url and img_url.startswith('http') and len(img_url) > 50:
                            candidates.append(Candidate(
                                url=img_url,
                                source=host_of(img_url),
                                thumb=None,
                                width=None,
                                height=None,
                                meta={"from": "korean_news"},
                                search_engine="korean_news"
                            ))
                            
        except Exception as e:
            print(f"한국 뉴스 사이트 검색 오류: {e}")
            continue
            
    return candidates[:max_results]

def search_wikimedia_images(query: str, max_results: int) -> List[Candidate]:
    """Wikimedia Commons 이미지 검색 (공개 API)"""
    candidates = []
    try:
        api_url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrlimit": max_results,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }
        response = SESSION.get(api_url, params=params, timeout=10)
        if response.status_code != 200:
            return candidates
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            imageinfo = page.get("imageinfo", [])
            if imageinfo:
                img_url = imageinfo[0].get("url", "")
                if img_url:
                    candidates.append(Candidate(
                        url=img_url,
                        source=host_of(img_url),
                        thumb=None,
                        width=None,
                        height=None,
                        meta={"from": "wikimedia"},
                        search_engine="wikimedia"
                    ))
                    if len(candidates) >= max_results:
                        break
    except Exception as e:
        print(f"Wikimedia 검색 오류: {e}")
    return candidates

# 기존 검색 엔진들 (개선된 버전)
def search_bing_images_improved(query: str, max_results: int) -> List[Candidate]:
    """개선된 Bing 이미지 검색"""
    candidates = []
    try:
        search_url = f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC2&first=1&tsc=ImageBasicHover"
        response = SESSION.get(search_url, timeout=12)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Bing의 JSON 데이터 추출
        for script in soup.find_all('script'):
            if script.string and 'murl' in script.string:
                try:
                    # JSON 데이터 파싱
                    matches = re.findall(r'{"murl":"([^"]+)".*?"purl":"([^"]*)".*?"w":(\d+).*?"h":(\d+)', script.string)
                    
                    for match in matches[:max_results]:
                        img_url, page_url, width, height = match
                        img_url = img_url.replace('\\/', '/').replace('\\u003d', '=')
                        
                        candidates.append(Candidate(
                            url=img_url,
                            source=host_of(page_url or img_url),
                            thumb=None,
                            width=int(width) if width.isdigit() else None,
                            height=int(height) if height.isdigit() else None,
                            meta={"from": "bing_improved"},
                            search_engine="bing"
                        ))
                        
                        if len(candidates) >= max_results:
                            break
                except:
                    continue
                    
    except Exception as e:
        print(f"Bing 개선 검색 오류: {e}")
        
    return candidates

# 향상된 이미지 검증
def fetch_and_validate_advanced(
    cand: Candidate,
    min_w: int, min_h: int, min_bytes: int,
    reference_faces: List[FaceData],
    face_threshold: float,
    blocked: Dict[str, tuple],
) -> Optional[ValidImage]:
    """고급 이미지 검증 - 다중 얼굴 매칭"""
    
    domain = cand.source or host_of(cand.url)
    
    # 차단 검사
    if cand.url in blocked["urls"] or any(domain == h or domain.endswith("." + h) for h in blocked["hosts"]):
        return None
    
    try:
        response = SESSION.get(cand.url, timeout=TIMEOUT, stream=True)
        ct = response.headers.get("Content-Type", "").lower()
        
        if response.status_code != 200 or "image" not in ct:
            return None
            
        raw = response.content
        if len(raw) < max(8000, min_bytes):  # 최소 크기 증가
            return None

        # 이미지 처리
        with Image.open(io.BytesIO(raw)) as img:
            w, h = img.size
            if w < min_w or h < min_h:
                return None
                
            # 이미지 품질 개선 (선명도 향상)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            fmt = (img.format or "JPEG").lower()
            focus_var = focus_variance(img)
            phash = str(imagehash.phash(img.convert("L")))

        # pHash 중복 검사
        try:
            new_hash = imagehash.hex_to_hash(phash)
            for blocked_hash in blocked["hashes"]:
                if new_hash - imagehash.hex_to_hash(blocked_hash) <= 4:  # 더 엄격한 중복 검사
                    return None
        except:
            pass

        # 고급 얼굴 매칭
        face_similarity = 0.0
        face_confidence = 0.0
        face_count = 0
        best_bbox = None
        
        if reference_faces and len(reference_faces) > 0:
            try:
                current_faces = extract_face_encodings_advanced(raw)
                
                if current_faces:
                    face_count = len(current_faces)
                    max_similarity = 0.0
                    best_face = None
                    
                    # 현재 이미지의 각 얼굴을 기준 얼굴들과 비교
                    for current_face in current_faces:
                        for ref_face in reference_faces:
                            similarity = compare_faces_advanced(
                                ref_face.encoding, 
                                current_face.encoding, 
                                "face_recognition" if FACE_LIBS["face_recognition"] else "cosine"
                            )
                            
                            # 가중치 적용 (기준 얼굴의 신뢰도 고려)
                            weighted_similarity = similarity * ref_face.confidence
                            
                            if weighted_similarity > max_similarity:
                                max_similarity = weighted_similarity
                                best_face = current_face
                                face_similarity = similarity
                                face_confidence = current_face.confidence
                                best_bbox = current_face.bbox
                    
                    # 얼굴 유사도 임계값 검사 (더 엄격)
                    if face_similarity < face_threshold:
                        return None
                        
                else:
                    # 얼굴이 전혀 감지되지 않은 경우 제외
                    if len(reference_faces) > 0:  # 기준 얼굴이 있다면 얼굴 없는 이미지는 제외
                        return None
                        
            except Exception as e:
                print(f"얼굴 인식 오류: {e}")
                # 얼굴 인식 실패 시 낮은 점수로 진행
                face_similarity = 0.2

        # 최종 스코어 계산 (개선된 공식)
        base_score = calculate_advanced_score(w, h, focus_var, domain, len(raw), face_similarity, face_confidence)
        
        return ValidImage(
            url=cand.url, source=domain, width=w, height=h,
            size_bytes=len(raw), fmt=fmt, phash=phash,
            focus_var=focus_var, score=base_score,
            face_similarity=face_similarity, face_confidence=face_confidence,
            face_count=face_count, thumb=cand.thumb, 
            search_engine=cand.search_engine, face_bbox=best_bbox
        )
        
    except Exception as e:
        print(f"검증 오류 {cand.url}: {e}")
        return None

def calculate_advanced_score(width: int, height: int, focus: float, domain: str, 
                           size_bytes: int, face_sim: float, face_conf: float) -> float:
    """고급 스코어 계산 - 얼굴 매칭 중심"""
    
    # 기본 품질 점수
    area = width * height
    resolution_score = min(area / 4_000_000, 1.0) * 0.25
    
    # 선명도 점수
    focus_score = min(focus / 250.0, 1.0) * 0.20
    
    # 파일 크기 점수
    size_score = min(size_bytes / (600 * 1024), 1.0) * 0.05
    
    # 도메인 점수
    domain_score = domain_weight(domain) * 0.15
    
    # 얼굴 매칭 점수 (가장 중요)
    face_score = (face_sim * 0.25) + (face_conf * 0.10)
    
    total_score = resolution_score + focus_score + size_score + domain_score + face_score
    
    return total_score

# 통합 검색 및 수집 함수
@st.cache_data(show_spinner=False, ttl=4*3600)
def search_and_collect_comprehensive(
    query: str, want: int,
    min_w: int, min_h: int, min_bytes: int,
    face_threshold: float, use_face_matching: bool,
    insta_handle: str,
    mem_version: int,
    blocked_urls: tuple, blocked_hosts: tuple, blocked_hashes: tuple,
) -> List[ValidImage]:
    """포괄적 이미지 검색 및 수집"""
    
    # 기준 얼굴 수집
    reference_faces = []
    if use_face_matching:
        reference_faces = collect_reference_faces(query, max_samples=5)
    
    all_candidates = []
    
    # 다양한 검색 쿼리
    search_queries = [
        f"{query} 프로필 고화질",
        f"{query} 공식 사진",
        f"{query} 포토 촬영",
        f"{query} portrait HD"
    ]
    
    # 각 검색 엔진에서 병렬 수집
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        for search_query in search_queries[:3]:  # 3개 쿼리만 사용
            # 주요 검색 엔진들
            futures.append(executor.submit(search_google_images_advanced, search_query, want//4))
            futures.append(executor.submit(search_bing_images_improved, search_query, want//4))
            
            # 추가 검색 소스들
            futures.append(executor.submit(search_yandex_images, search_query, want//6))
            futures.append(executor.submit(search_baidu_images, search_query, want//6))
            futures.append(executor.submit(search_pinterest_images, search_query, want//6))
            futures.append(executor.submit(search_korean_news_sites, search_query, want//6))
            futures.append(executor.submit(search_wikimedia_images, search_query, want//6))
            
            # DuckDuckGo (가능한 경우)
            if HAVE_DDG:
                try:
                    from duckduckgo_search import DDGS
                    futures.append(executor.submit(lambda q, n: [
                        Candidate(
                            url=r.get("image") or r.get("url"),
                            source=host_of(r.get("source") or r.get("url", "")),
                            thumb=r.get("thumbnail"),
                            width=r.get("width"), height=r.get("height"),
                            meta=r, search_engine="duckduckgo"
                        )
                        for r in DDGS().images(q, max_results=n)
                    ], search_query, want//6))
                except:
                    pass

        if insta_handle:
            futures.append(executor.submit(search_instagram_profile, insta_handle))
        
        # 결과 수집
        completed_futures = 0
        for future in as_completed(futures):
            try:
                candidates = future.result()
                all_candidates.extend(candidates)
                completed_futures += 1
                
                # 진행 상황 표시
                progress = int((completed_futures / len(futures)) * 50)
                st.progress(progress, text=f"검색 소스 {completed_futures}/{len(futures)} 완료...")
                
            except Exception as e:
                print(f"검색 오류: {e}")
                continue
    
    # URL 중복 제거
    seen_urls = set()
    unique_candidates = []
    for cand in all_candidates:
        if cand.url and cand.url not in seen_urls and len(cand.url) > 50:
            seen_urls.add(cand.url)
            unique_candidates.append(cand)
    
    if not unique_candidates:
        return []
    
    st.info(f"총 {len(unique_candidates)}개 후보에서 {len(set(c.search_engine for c in unique_candidates))}개 소스 활용")
    
    # 고급 이미지 검증
    valid_images = []
    total = len(unique_candidates)
    batch_size = 15  # 배치 크기 증가
    
    for i in range(0, total, batch_size):
        batch = unique_candidates[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=8) as executor:  # 워커 수 증가
            futures = [
                executor.submit(
                    fetch_and_validate_advanced,
                    cand, min_w, min_h, min_bytes,
                    reference_faces, face_threshold,
                    {"urls": blocked_urls, "hosts": blocked_hosts, "hashes": blocked_hashes}
                ) for cand in batch
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    valid_images.append(result)
        
        # 진행률 업데이트
        progress = int(50 + (min(i + batch_size, total) / total * 50))
        st.progress(progress, text=f"검증 완료: {min(i + batch_size, total)}/{total}")
    
    if not valid_images:
        return []
    
    # 고급 정렬 (얼굴 유사도 우선)
    valid_images.sort(key=lambda x: (
        x.face_similarity * 2,  # 얼굴 유사도 가중치 증가
        x.score,
        x.width * x.height,
        x.face_confidence
    ), reverse=True)
    
    # 고급 중복 제거 (pHash + 얼굴 위치 고려)
    final_images = []
    seen_hashes = set()
    
    for img in valid_images:
        try:
            current_hash = imagehash.hex_to_hash(img.phash)
            is_duplicate = False
            
            for seen_hash_str in seen_hashes:
                seen_hash = imagehash.hex_to_hash(seen_hash_str)
                if current_hash - seen_hash <= 5:  # 중복 임계값
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_hashes.add(img.phash)
                final_images.append(img)
                
        except:
            # 해시 처리 실패 시에도 추가 (안전장치)
            final_images.append(img)
    
    return final_images[:want]

# 검색 실행 함수
def run_advanced_search(name: str, opts: Dict[str, Any]):
    """고급 검색 실행"""
    blocked_data = {
        "urls": tuple(MEM.get("blocked_urls", [])),
        "hosts": tuple(MEM.get("blocked_hosts", [])),
        "hashes": tuple(MEM.get("blocked_hashes", [])),
        "version": int(MEM.get("version", 0))
    }
    
    with st.spinner("🔍 다중 소스에서 고품질 이미지 검색 및 얼굴 매칭 중..."):
        results = search_and_collect_comprehensive(
            query=name.strip(),
            want=int(opts["want"]),
            min_w=int(opts["min_w"]),
            min_h=int(opts["min_h"]),
            min_bytes=int(opts["min_bytes"]),
            face_threshold=float(opts["face_threshold"]),
            use_face_matching=bool(opts["use_face_matching"]),
            insta_handle=str(opts.get("insta_handle", "")),
            mem_version=blocked_data["version"],
            blocked_urls=blocked_data["urls"],
            blocked_hosts=blocked_data["hosts"],
            blocked_hashes=blocked_data["hashes"]
        )
    
    st.session_state["results"] = results
    st.session_state["query_name"] = name
    st.session_state["options"] = opts
    st.session_state["last_search_ok"] = bool(results)
    
    if results:
        face_matched = sum(1 for r in results if r.face_similarity > 0.5)
        sources = set(r.search_engine for r in results)
        high_confidence = sum(1 for r in results if r.face_similarity > 0.7)
        
        st.session_state["notice"] = (
            f"✅ 완료! {len(results)}개 결과 "
            f"(얼굴매칭: {face_matched}개, 고신뢰도: {high_confidence}개, "
            f"소스: {len(sources)}개)"
        )
    else:
        st.session_state["notice"] = "조건에 맞는 이미지를 찾지 못했습니다. 임계값을 낮추거나 다른 이름으로 시도해보세요."

# 이미지 다운로드
@st.cache_data(show_spinner=False, ttl=24*3600)
def get_image_bytes_cached(url: str) -> Optional[bytes]:
    try:
        response = SESSION.get(url, timeout=TIMEOUT)
        if response.status_code == 200 and "image" in response.headers.get("Content-Type", "").lower():
            return response.content
    except:
        return None
    return None

# ===== UI 시작 =====
st.title("🌟 연예인 이미지 파인더 v3.0 - AI 얼굴인식 + 글로벌 검색")
st.caption("🚀 최신 기능: 다중 기준얼굴 학습, 10개 검색엔진 통합, 고정밀 얼굴 매칭")

# 라이브러리 상태 표시
face_lib_status = []
if FACE_LIBS["face_recognition"]:
    face_lib_status.append("face_recognition ✅")
if FACE_LIBS["opencv"]:
    face_lib_status.append("OpenCV ✅")
if FACE_LIBS["mediapipe"]:
    face_lib_status.append("MediaPipe ✅")

if face_lib_status:
    st.success(f"얼굴 인식: {', '.join(face_lib_status)}")
else:
    st.error("⚠️ 얼굴 인식 라이브러리가 설치되지 않았습니다!")

# 사이드바 설정
with st.sidebar:
    st.subheader("🎯 고급 검색 설정")
    
    name = st.text_input(
        "연예인 이름",
        value=st.session_state.get("query_name", ""),
        placeholder="정확한 한글/영문 이름 입력",
        help="정확할수록 더 좋은 결과를 얻습니다"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        want = st.number_input(
            "결과 개수", min_value=30, max_value=500,
            value=int(st.session_state["options"].get("want", 100)),
            step=20, help="더 많은 결과 = 더 오래 걸림"
        )
    with col2:
        resolution_options = ["800x800", "1000x1000", "1200x1200", "1500x1500", "2000x2000"]
        min_res = st.selectbox(
            "최소 해상도",
            resolution_options,
            index=resolution_options.index(st.session_state["options"].get("min_res", "1200x1200"))
            if st.session_state["options"].get("min_res", "1200x1200") in resolution_options else 2
        )
    
    min_bytes_kb = st.slider(
        "최소 파일 크기(KB)", 100, 5000,
        int(st.session_state["options"].get("min_bytes", 300000) // 1024),
        50, help="고품질 이미지 확보"
    )
    
    st.markdown("---")
    st.subheader("🧠 AI 얼굴 인식")
    
    use_face_matching = st.checkbox(
        "고급 얼굴 매칭 사용",
        value=bool(st.session_state["options"].get("use_face_matching", bool(face_lib_status))),
        help="AI가 해당 인물의 얼굴만 선별",
        disabled=not bool(face_lib_status)
    )
    
    face_threshold = st.slider(
        "얼굴 유사도 임계값", 0.0, 1.0,
        float(st.session_state["options"].get("face_threshold", 0.65)),
        0.05, help="0.65 권장 (높을수록 엄격)"
    )
    
    if use_face_matching:
        st.info("🧠 다중 기준얼굴 자동 학습 + 가중치 매칭")
    
    st.markdown("---")
    st.subheader("🌍 검색 소스")
    st.info("""
    **10개 글로벌 검색엔진:**
    • Google (고품질)
    • Bing (다양성)  
    • Yandex (러시아)
    • Baidu (중국)
    • Pinterest (스타일)
    • 한국 뉴스사이트
    • DuckDuckGo
    • Naver, Daum
    • Wikimedia Commons
    """)

    st.markdown("---")
    st.subheader("📸 Instagram (선택)")
    insta_handle = st.text_input(
        "인스타그램 핸들",
        value=str(st.session_state["options"].get("insta_handle", "")),
        placeholder="@handle 또는 https://www.instagram.com/handle",
        help="공개 프로필의 프로필 이미지만 가져옵니다"
    )
    
    # 현재 옵션 정리
    min_w, min_h = map(int, min_res.split("x"))
    current_options = {
        "want": int(want),
        "min_res": min_res,
        "min_w": min_w,
        "min_h": min_h,
        "min_bytes": int(min_bytes_kb * 1024),
        "use_face_matching": bool(use_face_matching),
        "face_threshold": float(face_threshold),
        "insta_handle": normalize_instagram_handle(insta_handle),
    }
    
    st.markdown("---")
    search_disabled = not name.strip()
    
    if st.button(
        "🚀 AI 검색 시작",
        type="primary",
        use_container_width=True,
        disabled=search_disabled
    ):
        run_advanced_search(name, current_options)
        
    if st.session_state["results"]:
        if st.button("🔄 동일 조건 재검색", use_container_width=True):
            run_advanced_search(st.session_state["query_name"], st.session_state["options"])

# 알림 표시
if st.session_state["notice"]:
    if st.session_state["last_search_ok"]:
        st.success(st.session_state["notice"])
    else:
        st.warning(st.session_state["notice"])

# ===== 결과 표시 =====
results = st.session_state["results"]
query_name = st.session_state["query_name"]

if results:
    st.markdown("---")
    
    # 상세 통계
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("총 이미지", len(results))
    with col2:
        high_similarity = sum(1 for r in results if r.face_similarity > 0.7)
        st.metric("고신뢰도", f"{high_similarity}개")
    with col3:
        avg_similarity = np.mean([r.face_similarity for r in results if r.face_similarity > 0])
        st.metric("평균 유사도", f"{avg_similarity:.2f}" if avg_similarity > 0 else "N/A")
    with col4:
        avg_resolution = np.mean([r.width * r.height for r in results]) / 1_000_000
        st.metric("평균 해상도", f"{avg_resolution:.1f}MP")
    with col5:
        unique_sources = len(set(r.search_engine for r in results))
        st.metric("활용 소스", f"{unique_sources}개")
    
    st.subheader(f"🎨 {query_name} - AI 선별 결과")
    
    # 고급 필터
    with st.expander("🔧 고급 필터", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            min_face_sim_filter = st.slider("최소 얼굴 유사도", 0.0, 1.0, 0.0, 0.05)
            min_confidence_filter = st.slider("최소 얼굴 신뢰도", 0.0, 1.0, 0.0, 0.05)
        
        with filter_col2:
            min_score_filter = st.slider("최소 종합 점수", 0.0, 2.0, 0.0, 0.1)
            min_resolution_filter = st.slider("최소 픽셀(MP)", 0.0, 10.0, 0.0, 0.5)
        
        with filter_col3:
            source_filter = st.multiselect(
                "소스 선택",
                options=sorted(set(r.search_engine for r in results)),
                default=sorted(set(r.search_engine for r in results))
            )
    
    # 필터 적용
    filtered_results = [
        r for r in results
        if (r.face_similarity >= min_face_sim_filter and
            r.face_confidence >= min_confidence_filter and
            r.score >= min_score_filter and
            (r.width * r.height / 1_000_000) >= min_resolution_filter and
            r.search_engine in source_filter)
    ]
    
    if len(filtered_results) != len(results):
        st.info(f"필터 적용: {len(filtered_results)}/{len(results)}개 표시")
    
    # 이미지 그리드
    cols = st.columns(4, gap="small")
    memory_updated = False
    
    for i, img in enumerate(filtered_results):
        with cols[i % 4]:
            # 이미지 표시
            display_url = img.thumb or img.url
            
            # 신뢰도에 따른 테두리 색상
            confidence_color = "green" if img.face_similarity > 0.8 else "orange" if img.face_similarity > 0.6 else "red"
            
            st.image(
                display_url,
                caption=f"🔍{img.search_engine} | {img.width}×{img.height} | 유사도:{img.face_similarity:.2f}",
                use_container_width=True
            )
            
            # 상세 정보
            with st.expander("📊 상세 & 다운로드", expanded=False):
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.text(f"소스: {img.source}")
                    st.text(f"해상도: {img.width}×{img.height}")
                    st.text(f"크기: {img.size_bytes//1024}KB")
                    st.text(f"형식: {img.fmt.upper()}")
                
                with info_cols[1]:
                    st.text(f"얼굴유사도: {img.face_similarity:.3f}")
                    st.text(f"얼굴신뢰도: {img.face_confidence:.3f}")
                    st.text(f"종합점수: {img.score:.3f}")
                    st.text(f"선명도: {img.focus_var:.1f}")
                
                # 다운로드 버튼
                img_data = get_image_bytes_cached(img.url)
                if img_data:
                    filename = f"{query_name}_{i+1:03d}_{img.width}x{img.height}.{img.fmt}"
                    st.download_button(
                        "💾 다운로드",
                        data=img_data,
                        file_name=filename,
                        mime=f"image/{img.fmt}",
                        use_container_width=True,
                        key=f"dl_{i}"
                    )
                
                # 피드백
                feedback_cols = st.columns(3)
                with feedback_cols[0]:
                    if st.button("🚫", key=f"block_{i}", help="이 이미지 차단"):
                        if img.url not in MEM["blocked_urls"]:
                            MEM["blocked_urls"].append(img.url)
                        if img.phash not in MEM["blocked_hashes"]:
                            MEM["blocked_hashes"].append(img.phash)
                        save_memory(MEM)
                        memory_updated = True
                        st.success("차단됨")
                
                with feedback_cols[1]:
                    if st.button("🚫🌐", key=f"block_domain_{i}", help="도메인 차단"):
                        domain = img.source
                        if domain and domain not in MEM["blocked_hosts"]:
                            MEM["blocked_hosts"].append(domain)
                            if domain in MEM.get("preferred_hosts", []):
                                MEM["preferred_hosts"].remove(domain)
                            save_memory(MEM)
                            memory_updated = True
                            st.success("도메인 차단")
                
                with feedback_cols[2]:
                    if st.button("👍🌐", key=f"prefer_domain_{i}", help="도메인 선호"):
                        domain = img.source
                        if domain and domain not in MEM.get("preferred_hosts", []):
                            if "preferred_hosts" not in MEM:
                                MEM["preferred_hosts"] = []
                            MEM["preferred_hosts"].append(domain)
                            if domain in MEM.get("blocked_hosts", []):
                                MEM["blocked_hosts"].remove(domain)
                            save_memory(MEM)
                            memory_updated = True
                            st.success("선호 등록")
            
            # 원본 링크
            st.markdown(f"[🔗 원본보기]({img.url})")
    
    # 피드백 알림
    if memory_updated:
        st.toast("피드백이 저장되었습니다!", icon="✅")
    
    # 하단 액션
    st.markdown("---")
    action_cols = st.columns(4)
    
    with action_cols[0]:
        if st.button("🧹 결과 지우기"):
            for key in ["results", "notice", "last_search_ok"]:
                st.session_state[key] = [] if key == "results" else "" if key == "notice" else False
            st.rerun()
    
    with action_cols[1]:
        if st.button("🔄 피드백 반영 재검색"):
            if st.session_state.get("query_name"):
                run_advanced_search(st.session_state["query_name"], st.session_state["options"])
                st.rerun()
    
    with action_cols[2]:
        total_mb = sum(r.size_bytes for r in filtered_results) // (1024*1024)
        st.info(f"💾 {len(filtered_results)}개, {total_mb}MB")
    
    with action_cols[3]:
        if st.button("🎯 기준얼굴 재학습"):
            if query_name in MEM.get("celebrity_faces", {}):
                del MEM["celebrity_faces"][query_name]
                save_memory(MEM)
                st.success("기준얼굴 삭제됨. 재검색하면 새로 학습합니다.")

# ===== 설정 관리 =====
st.markdown("---")
with st.expander("⚙️ 시스템 관리"):
    st.json({
        "저장된_연예인": list(MEM.get("celebrity_faces", {}).keys()),
        "차단_URL": len(MEM.get("blocked_urls", [])),
        "차단_도메인": MEM.get("blocked_hosts", []),
        "선호_도메인": MEM.get("preferred_hosts", []),
        "메모리_버전": MEM.get("version", 0)
    })
    
    mgmt_cols = st.columns(3)
    with mgmt_cols[0]:
        if st.button("🗑️ 모든 차단정보 삭제"):
            MEM.update({
                "blocked_urls": [], "blocked_hosts": [], "blocked_hashes": [],
                "preferred_hosts": [], "version": MEM.get("version", 0)
            })
            save_memory(MEM)
            st.success("차단정보 초기화 완료")
    
    with mgmt_cols[1]:
        if st.button("👤 모든 기준얼굴 삭제"):
            MEM["celebrity_faces"] = {}
            save_memory(MEM)
            st.success("기준얼굴 데이터 초기화 완료")
    
    with mgmt_cols[2]:
        if st.button("🔄 캐시 무효화"):
            MEM["version"] = MEM.get("version", 0) + 100
            save_memory(MEM)
            st.success("캐시 강제 갱신됨")

# 하단 설치 가이드
st.markdown("---")
st.caption("""
**🔥 v3.0 주요 개선사항:**
- 🧠 다중 기준얼굴 자동 학습 (5개 각도/표정)
- 🌍 10개 글로벌 검색엔진 통합 (Google, Bing, Yandex, Baidu, Pinterest 등)
- 🎯 가중치 기반 고정밀 얼굴 매칭
- 📊 실시간 신뢰도 분석 및 필터링
- ⚡ 병렬 처리로 성능 대폭 향상

**필수 라이브러리 설치:**
```bash
pip install streamlit requests pillow beautifulsoup4 imagehash numpy python-dotenv
pip install face-recognition  # 최고 정밀도
pip install opencv-python mediapipe  # 보완용
pip install duckduckgo-search  # 추가 소스
```

**⚠️ 주의사항:** 개인 사용 목적이며, 저작권과 초상권을 존중해주세요.
""")

if __name__ == "__main__":
    st.markdown("---")
    st.success("🚀 연예인 이미지 파인더 v3.0이 실행 중입니다!")
    if not any(FACE_LIBS.values()):
        st.error("""
        ❌ 얼굴 인식 기능을 사용하려면 다음 중 하나를 설치하세요:
        
        **최고 성능 (권장):**
        ```bash
        pip install face-recognition
        ```
        
        **기본 성능:**
        ```bash  
        pip install opencv-python
        ```
        
        **모바일 최적화:**
        ```bash
        pip install mediapipe
        ```
        """)
    else:
        available_libs = [name for name, lib in FACE_LIBS.items() if lib]
        st.info(f"✅ 사용 가능한 얼굴 인식: {', '.join(available_libs)}")
