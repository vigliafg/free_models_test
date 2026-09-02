#!/usr/bin/env python3
"""
Script per testare tutti i modelli gratuiti di OpenRouter, NVIDIA NIM, Mistral e Groq.
Connette ai provider, cerca i modelli free, li chiama e fa un reporting unificato.

Uso:
    python3 test_free_models.py                     # test singolo (default)
    python3 test_free_models.py --benchmark         # benchmark: 3 round, pausa 10s,
                                                    # media ms per modello valido
    python3 test_free_models.py --benchmark --providers groq,mistral --rounds 3 --pause 10
"""

import os
import re
import json
import time
import sys
import argparse
import statistics
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


@dataclass
class ModelResult:
    model_id: str
    model_name: str
    provider: str  # "openrouter", "nvidia_nim", "mistral", "groq"
    success: bool
    response_time_ms: float
    response_text: str
    error: Optional[str] = None
    tokens_used: Optional[int] = None


@dataclass
class TestConfig:
    test_prompt: str = "Scrivi una breve frase in italiano su come funziona un LLM."
    max_tokens: int = 100
    temperature: float = 0.7
    timeout: int = 30
    max_concurrent: int = 3


class BaseTester:
    """Classe base per i tester dei provider."""
    
    def __init__(self, api_key: str, config: TestConfig = None):
        self.api_key = api_key
        self.config = config or TestConfig()
        self.session = requests.Session()
    
    def call_model(self, model: dict) -> ModelResult:
        raise NotImplementedError
    
    def test_all_models(self, models: List[dict]) -> List[ModelResult]:
        """Testa tutti i modelli in parallelo (con limite di concorrenza)."""
        results = []
        print(f"\n🚀 Testando {len(models)} modelli {self.provider_name} (max {self.config.max_concurrent} concorrenti)...\n")
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            future_to_model = {
                executor.submit(self.call_model, model): model 
                for model in models
            }
            
            for i, future in enumerate(as_completed(future_to_model), 1):
                result = future.result()
                results.append(result)
                
                status = "✅" if result.success else "❌"
                print(f"  [{i}/{len(models)}] {status} {result.model_name} ({result.model_id}) - {result.response_time_ms:.0f}ms")
                if not result.success:
                    print(f"      Errore: {result.error}")
        
        return results


class OpenRouterTester(BaseTester):
    BASE_URL = "https://openrouter.ai/api/v1"
    provider_name = "OpenRouter"
    
    def __init__(self, api_key: str, config: TestConfig = None):
        super().__init__(api_key, config)
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vigliafg/openrouter-tester",
            "X-Title": "OpenRouter Free Models Tester"
        })
    
    def fetch_models(self) -> List[dict]:
        """Scarica la lista di tutti i modelli da OpenRouter."""
        print("🔍 Recupero lista modelli da OpenRouter...")
        response = self.session.get(f"{self.BASE_URL}/models", timeout=30)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", [])
        print(f"   Trovati {len(models)} modelli totali")
        return models
    
    def filter_free_models(self, models: List[dict]) -> List[dict]:
        """Filtra solo i modelli gratuiti (prompt=0, completion=0)."""
        free_models = [
            m for m in models 
            if m.get("pricing", {}).get("prompt") == "0" 
            and m.get("pricing", {}).get("completion") == "0"
        ]
        print(f"   Modelli gratuiti: {len(free_models)}")
        return free_models
    
    def call_model(self, model: dict) -> ModelResult:
        """Chiama un singolo modello e misura il tempo di risposta."""
        model_id = model["id"]
        model_name = model.get("name", model_id)
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": self.config.test_prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                response_text = (message.get("content") or "").strip()
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens")
                
                return ModelResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider="openrouter",
                    success=True,
                    response_time_ms=round(elapsed_ms, 2),
                    response_text=response_text[:500],
                    tokens_used=tokens_used
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                return ModelResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider="openrouter",
                    success=False,
                    response_time_ms=round(elapsed_ms, 2),
                    response_text="",
                    error=error_msg
                )
                
        except requests.exceptions.Timeout:
            elapsed_ms = (time.time() - start_time) * 1000
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                provider="openrouter",
                success=False,
                response_time_ms=round(elapsed_ms, 2),
                response_text="",
                error=f"Timeout ({self.config.timeout}s)"
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                provider="openrouter",
                success=False,
                response_time_ms=round(elapsed_ms, 2),
                response_text="",
                error=str(e)
            )


class NvidiaNimTester(BaseTester):
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    provider_name = "NVIDIA NIM"
    
    def __init__(self, api_key: str, config: TestConfig = None):
        super().__init__(api_key, config)
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def fetch_models(self) -> List[dict]:
        """Scarica la lista di tutti i modelli da NVIDIA NIM."""
        print("🔍 Recupero lista modelli da NVIDIA NIM...")
        response = self.session.get(f"{self.BASE_URL}/models", timeout=30)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", [])
        print(f"   Trovati {len(models)} modelli totali (tutti gratuiti su free tier)")
        return models
    
    def filter_free_models(self, models: List[dict]) -> List[dict]:
        """Su NVIDIA NIM free tier, tutti i modelli restituiti sono gratuiti."""
        return models
    
    def call_model(self, model: dict) -> ModelResult:
        """Chiama un singolo modello NVIDIA NIM e misura il tempo di risposta."""
        model_id = model["id"]
        model_name = model.get("name", model_id) if "name" in model else model_id
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": self.config.test_prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                response_text = (message.get("content") or "").strip()
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens")
                
                return ModelResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider="nvidia_nim",
                    success=True,
                    response_time_ms=round(elapsed_ms, 2),
                    response_text=response_text[:500],
                    tokens_used=tokens_used
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                return ModelResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider="nvidia_nim",
                    success=False,
                    response_time_ms=round(elapsed_ms, 2),
                    response_text="",
                    error=error_msg
                )
                
        except requests.exceptions.Timeout:
            elapsed_ms = (time.time() - start_time) * 1000
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                provider="nvidia_nim",
                success=False,
                response_time_ms=round(elapsed_ms, 2),
                response_text="",
                error=f"Timeout ({self.config.timeout}s)"
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                provider="nvidia_nim",
                success=False,
                response_time_ms=round(elapsed_ms, 2),
                response_text="",
                error=str(e)
            )


class MistralTester(BaseTester):
    BASE_URL = "https://api.mistral.ai/v1"
    provider_name = "Mistral"
    
    def __init__(self, api_key: str, config: TestConfig = None):
        super().__init__(api_key, config)
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def fetch_models(self) -> List[dict]:
        """Scarica la lista di tutti i modelli da Mistral."""
        print("🔍 Recupero lista modelli da Mistral...")
        response = self.session.get(f"{self.BASE_URL}/models", timeout=30)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", [])
        print(f"   Trovati {len(models)} modelli totali")
        return models
    
    def filter_free_models(self, models: List[dict]) -> List[dict]:
        """Tier Experiment di Mistral: scarta i modelli non-chat (embed, OCR,
        moderation, TTS, audio/realtime) e deduplica gli alias mantenendo un
        solo id per modello sottostante (preferendo l'id canonico id == name)."""
        skip_patterns = (
            "embed",        # mistral-embed, codestral-embed
            "ocr",          # mistral-ocr*
            "moderation",   # mistral-moderation*
            "tts",          # voxtral-*-tts
            "realtime",     # voxtral-*-transcribe-realtime
            "transcribe",   # variante realtime
            "voxtral",      # modelli audio in generale
        )
        candidates = [
            m for m in models
            if not any(
                p in m.get("id", "").lower() or p in m.get("name", "").lower()
                for p in skip_patterns
            )
        ]
        # Gli alias condividono lo stesso campo "name" con id diversi:
        # tieni una sola voce per name, preferendo l'id canonico (id == name)
        # e, a parita', l'id piu' corto (l'id datato tipo "mistral-small-2603").
        best = {}
        for m in candidates:
            mid = m.get("id", "")
            name = m.get("name") or mid
            rank = (0 if mid == name else 1, len(mid))
            cur = best.get(name)
            if cur is None or rank < cur[0]:
                best[name] = (rank, m)
        filtered = [v[1] for v in best.values()]
        print(f"   Filtrati {len(models) - len(filtered)} modelli (non-chat o alias duplicati)")
        return filtered
    
    def call_model(self, model: dict) -> ModelResult:
        """Chiama un singolo modello Mistral e misura il tempo di risposta."""
        model_id = model["id"]
        model_name = model.get("name", model_id) if "name" in model else model_id
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": self.config.test_prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                response_text = (message.get("content") or "").strip()
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens")
                
                return ModelResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider="mistral",
                    success=True,
                    response_time_ms=round(elapsed_ms, 2),
                    response_text=response_text[:500],
                    tokens_used=tokens_used
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                return ModelResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider="mistral",
                    success=False,
                    response_time_ms=round(elapsed_ms, 2),
                    response_text="",
                    error=error_msg
                )
                
        except requests.exceptions.Timeout:
            elapsed_ms = (time.time() - start_time) * 1000
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                provider="mistral",
                success=False,
                response_time_ms=round(elapsed_ms, 2),
                response_text="",
                error=f"Timeout ({self.config.timeout}s)"
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                provider="mistral",
                success=False,
                response_time_ms=round(elapsed_ms, 2),
                response_text="",
                error=str(e)
            )


class GroqTester(BaseTester):
    BASE_URL = "https://api.groq.com/openai/v1"
    provider_name = "Groq"
    
    def __init__(self, api_key: str, config: TestConfig = None):
        super().__init__(api_key, config)
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def fetch_models(self) -> List[dict]:
        """Scarica la lista di tutti i modelli da Groq."""
        print("🔍 Recupero lista modelli da Groq...")
        response = self.session.get(f"{self.BASE_URL}/models", timeout=30)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", [])
        print(f"   Trovati {len(models)} modelli totali (tutti gratuiti su free tier)")
        return models
    
    def filter_free_models(self, models: List[dict]) -> List[dict]:
        """Su Groq free tier, tutti i modelli restituiti sono gratuiti."""
        return models
    
    def call_model(self, model: dict) -> ModelResult:
        """Chiama un singolo modello Groq e misura il tempo di risposta."""
        model_id = model["id"]
        model_name = model.get("name", model_id) if "name" in model else model_id
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": self.config.test_prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                response_text = (message.get("content") or "").strip()
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens")
                
                return ModelResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider="groq",
                    success=True,
                    response_time_ms=round(elapsed_ms, 2),
                    response_text=response_text[:500],
                    tokens_used=tokens_used
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                return ModelResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider="groq",
                    success=False,
                    response_time_ms=round(elapsed_ms, 2),
                    response_text="",
                    error=error_msg
                )
                
        except requests.exceptions.Timeout:
            elapsed_ms = (time.time() - start_time) * 1000
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                provider="groq",
                success=False,
                response_time_ms=round(elapsed_ms, 2),
                response_text="",
                error=f"Timeout ({self.config.timeout}s)"
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                provider="groq",
                success=False,
                response_time_ms=round(elapsed_ms, 2),
                response_text="",
                error=str(e)
            )


class UnifiedReporter:
    """Genera report unificati per entrambi i provider."""
    
    def __init__(self, config: TestConfig):
        self.config = config
    
    def generate_report(self, all_results: List[ModelResult]) -> str:
        """Genera un report testuale dei risultati per tutti i provider."""
        # Separa per provider
        or_results = [r for r in all_results if r.provider == "openrouter"]
        nim_results = [r for r in all_results if r.provider == "nvidia_nim"]
        mistral_results = [r for r in all_results if r.provider == "mistral"]
        groq_results = [r for r in all_results if r.provider == "groq"]
        
        lines = []
        lines.append("=" * 70)
        lines.append("REPORT TEST MODELLI GRATUITI - OPENROUTER + NVIDIA NIM + MISTRAL + GROQ")
        lines.append("=" * 70)
        lines.append(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Prompt di test: \"{self.config.test_prompt}\"")
        lines.append(f"Max token: {self.config.max_tokens}, Temperature: {self.config.temperature}")
        lines.append("")
        
        # Report OpenRouter
        if or_results:
            lines.append(self._provider_section("OPENROUTER", or_results))
        
        # Report NVIDIA NIM
        if nim_results:
            lines.append(self._provider_section("NVIDIA NIM", nim_results))
        
        # Report Mistral
        if mistral_results:
            lines.append(self._provider_section("MISTRAL", mistral_results))
        
        # Report Groq
        if groq_results:
            lines.append(self._provider_section("GROQ", groq_results))
        
        # Summary comparativo
        lines.append(self._comparative_summary(or_results, nim_results, mistral_results, groq_results))
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def _provider_section(self, provider_name: str, results: List[ModelResult]) -> str:
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        successful.sort(key=lambda x: x.response_time_ms)
        
        lines = []
        lines.append(f"\n{'=' * 70}")
        lines.append(f"PROVIDER: {provider_name}")
        lines.append(f"{'=' * 70}")
        lines.append(f"📊 RIEPILOGO {provider_name}:")
        lines.append(f"   Totale modelli testati: {len(results)}")
        lines.append(f"   ✅ Riusciti: {len(successful)}")
        lines.append(f"   ❌ Falliti: {len(failed)}")
        lines.append("")
        
        if successful:
            avg_time = sum(r.response_time_ms for r in successful) / len(successful)
            lines.append(f"⚡ PERFORMANCE (modelli riusciti):")
            lines.append(f"   Tempo medio: {avg_time:.0f}ms")
            lines.append(f"   Più veloce: {successful[0].model_name} ({successful[0].response_time_ms:.0f}ms)")
            lines.append(f"   Più lento: {successful[-1].model_name} ({successful[-1].response_time_ms:.0f}ms)")
            lines.append("")
            
            lines.append(f"📋 DETTAGLIO MODELLI RIUSCITI (ordinati per velocità):")
            lines.append("-" * 70)
            for i, r in enumerate(successful, 1):
                lines.append(f"  {i:2d}. {r.model_name}")
                lines.append(f"       ID: {r.model_id}")
                lines.append(f"       Tempo: {r.response_time_ms:.0f}ms | Token: {r.tokens_used or 'N/A'}")
                lines.append(f"       Risposta: {r.response_text[:150]}{'...' if len(r.response_text) > 150 else ''}")
                lines.append("")
        
        if failed:
            lines.append(f"❌ MODELLI FALLITI:")
            lines.append("-" * 70)
            for i, r in enumerate(failed, 1):
                lines.append(f"  {i:2d}. {r.model_name} ({r.model_id})")
                lines.append(f"       Errore: {r.error}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _comparative_summary(self, or_results: List[ModelResult], nim_results: List[ModelResult], 
                           mistral_results: List[ModelResult], groq_results: List[ModelResult]) -> str:
        lines = []
        lines.append(f"\n{'=' * 70}")
        lines.append("RIEPILOGO COMPARATIVO")
        lines.append(f"{'=' * 70}")
        
        or_success = len([r for r in or_results if r.success])
        or_total = len(or_results)
        nim_success = len([r for r in nim_results if r.success])
        nim_total = len(nim_results)
        mistral_success = len([r for r in mistral_results if r.success])
        mistral_total = len(mistral_results)
        groq_success = len([r for r in groq_results if r.success])
        groq_total = len(groq_results)
        
        lines.append(f"OpenRouter:     {or_success}/{or_total} riusciti")
        lines.append(f"NVIDIA NIM:     {nim_success}/{nim_total} riusciti")
        lines.append(f"Mistral:        {mistral_success}/{mistral_total} riusciti")
        lines.append(f"Groq:           {groq_success}/{groq_total} riusciti")
        lines.append("")
        
        # Migliori per velocità per provider
        or_successful = [r for r in or_results if r.success]
        nim_successful = [r for r in nim_results if r.success]
        mistral_successful = [r for r in mistral_results if r.success]
        groq_successful = [r for r in groq_results if r.success]
        
        if or_successful:
            or_fastest = min(or_successful, key=lambda x: x.response_time_ms)
            lines.append(f"🏃 OpenRouter più veloce: {or_fastest.model_name} ({or_fastest.response_time_ms:.0f}ms)")
        
        if nim_successful:
            nim_fastest = min(nim_successful, key=lambda x: x.response_time_ms)
            lines.append(f"🏃 NVIDIA NIM più veloce: {nim_fastest.model_name} ({nim_fastest.response_time_ms:.0f}ms)")
        
        if mistral_successful:
            mistral_fastest = min(mistral_successful, key=lambda x: x.response_time_ms)
            lines.append(f"🏃 Mistral più veloce: {mistral_fastest.model_name} ({mistral_fastest.response_time_ms:.0f}ms)")
        
        if groq_successful:
            groq_fastest = min(groq_successful, key=lambda x: x.response_time_ms)
            lines.append(f"🏃 Groq più veloce: {groq_fastest.model_name} ({groq_fastest.response_time_ms:.0f}ms)")
        
        # Classifica generale per velocità
        all_successful = or_successful + nim_successful + mistral_successful + groq_successful
        if all_successful:
            all_successful.sort(key=lambda x: x.response_time_ms)
            lines.append("")
            lines.append("🏁 CLASSIFICA GENERALE PER VELOCITÀ (tutti i provider):")
            for i, r in enumerate(all_successful, 1):
                lines.append(f"  {i:2d}. {r.provider.upper():12s} | {r.model_name:<30s} | {r.response_time_ms:>7.0f}ms")
        
        lines.append("")
        return "\n".join(lines)
    
    def save_json_report(self, all_results: List[ModelResult], filepath: str):
        """Salva il report completo in JSON."""
        or_results = [r for r in all_results if r.provider == "openrouter"]
        nim_results = [r for r in all_results if r.provider == "nvidia_nim"]
        mistral_results = [r for r in all_results if r.provider == "mistral"]
        groq_results = [r for r in all_results if r.provider == "groq"]
        
        data = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "config": asdict(self.config),
            "summary": {
                "openrouter": {
                    "total": len(or_results),
                    "successful": len([r for r in or_results if r.success]),
                    "failed": len([r for r in or_results if not r.success])
                },
                "nvidia_nim": {
                    "total": len(nim_results),
                    "successful": len([r for r in nim_results if r.success]),
                    "failed": len([r for r in nim_results if not r.success])
                },
                "mistral": {
                    "total": len(mistral_results),
                    "successful": len([r for r in mistral_results if r.success]),
                    "failed": len([r for r in mistral_results if not r.success])
                },
                "groq": {
                    "total": len(groq_results),
                    "successful": len([r for r in groq_results if r.success]),
                    "failed": len([r for r in groq_results if not r.success])
                },
                "total": len(all_results),
                "total_successful": len([r for r in all_results if r.success]),
                "total_failed": len([r for r in all_results if not r.success])
            },
            "results": [asdict(r) for r in all_results]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Report JSON salvato in: {filepath}")


def _scrape_nvidia_free_ids() -> set:
    """Scrapa la pagina NVIDIA filtrata 'Free Endpoint' (build.nvidia.com) ed
    estrae gli id dei modelli con il badge; fallback su cache locale."""
    cache = "/tmp/nvidia_free_endpoints.json"
    url = "https://build.nvidia.com/models?filters=nimType%3Anim_type_preview"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    free = set()
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        text = resp.text
        # payload RSC escapato: \"Free Endpoint\" ... \"href\":\"/org/model\"
        for m in re.finditer(r'\\"Free Endpoint\\"', text):
            nxt = re.search(r'\\"href\\":\\"/([a-z0-9_.-]+/[a-z0-9_.-]+)\\"', text[m.end():m.end() + 800], re.I)
            if nxt:
                free.add(nxt.group(1))
        # HTML normale: >Free Endpoint< ... href="/org/model"
        for m in re.finditer(r'>Free Endpoint<', text):
            nxt = re.search(r'href="/([a-z0-9_.-]+/[a-z0-9_.-]+)"', text[m.end():m.end() + 800], re.I)
            if nxt:
                free.add(nxt.group(1))
    except Exception as e:
        print(f"[nvidia] scrape fallito: {e}")
    if free:
        try:
            with open(cache, "w") as f:
                json.dump(sorted(free), f, indent=2)
        except Exception:
            pass
        print(f"[nvidia] scraping OK: {len(free)} modelli 'Free Endpoint'")
        return free
    try:
        with open(cache) as f:
            cached = set(json.load(f))
        print(f"[nvidia] uso cache: {len(cached)} modelli")
        return cached
    except Exception:
        return set()


_NVIDIA_SKIP = re.compile(r"embed|voicechat|video|speaker|realtime", re.I)


def _restrict_nvidia_to_free(models: List[dict]) -> List[dict]:
    """Limita il catalogo NVIDIA ai modelli marcati 'Free Endpoint' sulla pagina
    ufficiale, escludendo le famiglie non testabili via chat/completions."""
    free_ids = _scrape_nvidia_free_ids()
    if not free_ids:
        print("[nvidia] ATTENZIONE: lista free non disponibile, uso tutto il catalogo")
        return models
    kept = [m for m in models
            if m.get("id") in free_ids and not _NVIDIA_SKIP.search(m.get("id", ""))]
    print(f"[nvidia] {len(kept)}/{len(models)} modelli sono 'Free Endpoint' e testabili via chat")
    return kept


def run_benchmark(providers: List[str], rounds: int, pause_s: int, config: TestConfig):
    """Benchmark: ogni modello valido e' testato `rounds` volte con `pause_s`
    tra i round; media/mediana/stdev dei ms. I provider girano in parallelo
    (quote separate), la concorrenza interna resta config.max_concurrent."""
    keys = {
        "openrouter": (os.environ.get("OPENROUTER_API_KEY"), OpenRouterTester),
        "nvidia_nim": (os.environ.get("NVIDIA_API_KEY"), NvidiaNimTester),
        "mistral": (os.environ.get("MISTRAL_API_KEY"), MistralTester),
        "groq": (os.environ.get("GROQ_API_KEY"), GroqTester),
    }
    active = {p: keys[p] for p in providers if p in keys and keys[p][0]}
    missing = [p for p in providers if p not in active]
    if missing:
        print(f"Provider senza chiave, saltati: {', '.join(missing)}")
    if not active:
        print("❌ Nessun provider attivo")
        return

    out = {}

    def bench_one(pname, key, tester_cls):
        try:
            tester = tester_cls(key, config)
            models = tester.fetch_models()
            free = tester.filter_free_models(models)
            if pname == "nvidia_nim":
                free = _restrict_nvidia_to_free(free)
            print(f"[{pname}] {len(free)} modelli da benchmarkare", flush=True)
            timings, fails = {}, {}
            for rnd in range(1, rounds + 1):
                if rnd > 1:
                    print(f"[{pname}] pausa {pause_s}s prima del round {rnd}...", flush=True)
                    time.sleep(pause_s)
                targets = free if rnd == 1 else [m for m in free if m["id"] in timings]
                if not targets:
                    continue
                results = tester.test_all_models(targets)
                for r in results:
                    if r.success:
                        timings.setdefault(r.model_id, []).append(r.response_time_ms)
                    else:
                        fails[r.model_id] = r.error
                ok = sum(1 for r in results if r.success)
                print(f"[{pname}] round {rnd}: {ok}/{len(results)} ok", flush=True)
            out[pname] = {"timings": timings, "fails": fails, "total": len(free)}
        except Exception as e:
            print(f"[{pname}] ERRORE provider: {e}", flush=True)
            out[pname] = {"timings": {}, "fails": {"__provider__": str(e)}, "total": 0}

    with ThreadPoolExecutor(max_workers=len(active)) as ex:
        futs = {ex.submit(bench_one, p, k, c): p for p, (k, c) in active.items()}
        for f in as_completed(futs):
            print(f"=== completato: {futs[f]} ===", flush=True)

    # Report medie
    print("\n" + "=" * 78)
    print(f"BENCHMARK {rounds} ROUND (pausa {pause_s}s) - MEDIA MSEC PER MODELLO VALIDO")
    print("=" * 78)
    all_rows = []
    for pname in providers:
        data = out.get(pname)
        if not data:
            continue
        timings, fails = data["timings"], data["fails"]
        print("\n" + "-" * 78)
        print(f"PROVIDER: {pname.upper()}   (modelli validi: {len(timings)}/{data['total']})")
        print("-" * 78)
        if not timings:
            print("   nessun modello valido")
            if "__provider__" in fails:
                print(f"   errore provider: {fails['__provider__']}")
            continue
        rows = []
        for mid, ms_list in timings.items():
            avg = statistics.mean(ms_list)
            med = statistics.median(ms_list)
            stdev = statistics.stdev(ms_list) if len(ms_list) > 1 else 0.0
            rows.append((avg, mid, ms_list, med, stdev))
        rows.sort()
        for avg, mid, ms_list, med, stdev in rows:
            print(f"  {avg:7.0f}ms  mediana {med:6.0f}  stdev {stdev:6.0f}  {mid[:48]:<48s} [{' ,'.join('%.0f' % x for x in ms_list)}]")
            all_rows.append((avg, pname, mid, len(ms_list)))
        bad = {k: v for k, v in fails.items() if k != "__provider__"}
        if bad:
            print("\n  Scartati (falliti nel round 1):")
            for k, v in sorted(bad.items()):
                print(f"     {k:<45s} {(v or '?')[:85].replace(chr(10), ' ')}")

    all_rows.sort()
    print("\n" + "=" * 78)
    print("CLASSIFICA GLOBALE (media, ms):")
    for i, (avg, pname, mid, n) in enumerate(all_rows[:25], 1):
        print(f"  {i:2d}. {pname:<12s} {mid[:48]:<48s} {avg:7.0f}ms ({n}/{rounds} round)")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Test/benchmark dei modelli LLM gratuiti")
    parser.add_argument("--benchmark", action="store_true",
                        help="modalita' benchmark: N round con pausa, media ms per modello valido")
    parser.add_argument("--providers", default="groq,mistral,openrouter,nvidia_nim",
                        help="provider da usare, separati da virgola (solo benchmark)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="numero di round nel benchmark (default 3)")
    parser.add_argument("--pause", type=int, default=10,
                        help="pausa in secondi tra i round (default 10)")
    args = parser.parse_args()

    if args.benchmark:
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        config = TestConfig(
            test_prompt="Scrivi una breve frase in italiano su come funziona un LLM.",
            max_tokens=100, temperature=0.7, timeout=30, max_concurrent=3,
        )
        run_benchmark(providers, args.rounds, args.pause, config)
        return

    # Leggi API keys dall'ambiente
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    
    if not openrouter_key and not nvidia_key and not mistral_key and not groq_key:
        print("❌ Errore: Nessuna API key configurata")
        print("   Imposta OPENROUTER_API_KEY e/o NVIDIA_API_KEY e/o MISTRAL_API_KEY e/o GROQ_API_KEY")
        sys.exit(1)
    
    # Configurazione test
    config = TestConfig(
        test_prompt="Scrivi una breve frase in italiano su come funziona un LLM.",
        max_tokens=100,
        temperature=0.7,
        timeout=30,
        max_concurrent=3
    )
    
    all_results = []
    reporter = UnifiedReporter(config)
    
    try:
        # Test OpenRouter se key presente
        if openrouter_key:
            print("\n" + "=" * 70)
            print("TEST OPENROUTER")
            print("=" * 70)
            or_tester = OpenRouterTester(openrouter_key, config)
            or_models = or_tester.fetch_models()
            or_free = or_tester.filter_free_models(or_models)
            
            if or_free:
                print(f"\n📋 Modelli gratuiti OpenRouter da testare:")
                for i, m in enumerate(or_free, 1):
                    print(f"   {i:2d}. {m.get('name', m['id'])} ({m['id']})")
                or_results = or_tester.test_all_models(or_free)
                all_results.extend(or_results)
            else:
                print("⚠️ Nessun modello gratuito trovato su OpenRouter!")
        
        # Test NVIDIA NIM se key presente
        if nvidia_key:
            print("\n" + "=" * 70)
            print("TEST NVIDIA NIM")
            print("=" * 70)
            nim_tester = NvidiaNimTester(nvidia_key, config)
            nim_models = nim_tester.fetch_models()
            nim_free = nim_tester.filter_free_models(nim_models)
            
            if nim_free:
                print(f"\n📋 Modelli gratuiti NVIDIA NIM da testare:")
                for i, m in enumerate(nim_free, 1):
                    name = m.get("name", m["id"])
                    print(f"   {i:2d}. {name} ({m['id']})")
                nim_results = nim_tester.test_all_models(nim_free)
                all_results.extend(nim_results)
            else:
                print("⚠️ Nessun modello trovato su NVIDIA NIM!")
        
        # Test Mistral se key presente
        if mistral_key:
            print("\n" + "=" * 70)
            print("TEST MISTRAL")
            print("=" * 70)
            mistral_tester = MistralTester(mistral_key, config)
            mistral_models = mistral_tester.fetch_models()
            mistral_free = mistral_tester.filter_free_models(mistral_models)
            
            if mistral_free:
                print(f"\n📋 Modelli gratuiti Mistral da testare:")
                for i, m in enumerate(mistral_free, 1):
                    name = m.get("name", m["id"])
                    print(f"   {i:2d}. {name} ({m['id']})")
                mistral_results = mistral_tester.test_all_models(mistral_free)
                all_results.extend(mistral_results)
            else:
                print("⚠️ Nessun modello trovato su Mistral!")
        
        # Test Groq se key presente
        if groq_key:
            print("\n" + "=" * 70)
            print("TEST GROQ")
            print("=" * 70)
            groq_tester = GroqTester(groq_key, config)
            groq_models = groq_tester.fetch_models()
            groq_free = groq_tester.filter_free_models(groq_models)
            
            if groq_free:
                print(f"\n📋 Modelli gratuiti Groq da testare:")
                for i, m in enumerate(groq_free, 1):
                    name = m.get("name", m["id"])
                    print(f"   {i:2d}. {name} ({m['id']})")
                groq_results = groq_tester.test_all_models(groq_free)
                all_results.extend(groq_results)
            else:
                print("⚠️ Nessun modello trovato su Groq!")
        
        if not all_results:
            print("\n⚠️ Nessun modello testato!")
            sys.exit(0)
        
        # Genera e stampa report unificato
        report = reporter.generate_report(all_results)
        print("\n" + report)
        
        # Salva JSON
        json_path = f"/home/vigliafg/free_models_test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        reporter.save_json_report(all_results, json_path)
        
        # Exit code basato su successo
        failed_count = len([r for r in all_results if not r.success])
        sys.exit(0 if failed_count == 0 else 1)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore di rete: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()