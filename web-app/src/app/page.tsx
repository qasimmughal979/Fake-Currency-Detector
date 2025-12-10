'use client';

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, ImageIcon, Loader2, CheckCircle2, XCircle, Box, Lock, Search, Settings, Sparkles, Shield, Zap, ScanLine } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';
import { GlowingEffect } from '@/components/ui/glowing-effect';
import { HeroGeometric } from '@/components/ui/shape-landing-hero';

// Utility for class merging
function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

// Grid Item Component for Features
interface GridItemProps {
  area: string;
  icon: React.ReactNode;
  title: string;
  description: React.ReactNode;
}

const GridItem = ({ area, icon, title, description }: GridItemProps) => {
  return (
    <li className={cn("min-h-[14rem] list-none", area)}>
      <div className="relative h-full rounded-[1.25rem] border-[0.75px] border-slate-800 p-2 md:rounded-[1.5rem] md:p-3">
        <GlowingEffect
          spread={40}
          glow={true}
          disabled={false}
          proximity={64}
          inactiveZone={0.01}
          borderWidth={3}
        />
        <div className="relative flex h-full flex-col justify-between gap-6 overflow-hidden rounded-xl border-[0.75px] border-slate-800 bg-slate-900/50 p-6 shadow-sm dark:shadow-[0px_0px_27px_0px_rgba(45,45,45,0.3)] md:p-6 backdrop-blur-sm">
          <div className="relative flex flex-1 flex-col justify-between gap-3">
            <div className="w-fit rounded-lg border-[0.75px] border-slate-700 bg-slate-800 p-2 text-cyan-400">
              {icon}
            </div>
            <div className="space-y-3">
              <h3 className="pt-0.5 text-xl leading-[1.375rem] font-semibold font-sans tracking-[-0.04em] md:text-2xl md:leading-[1.875rem] text-balance text-slate-100">
                {title}
              </h3>
              <h2 className="[&_b]:md:font-semibold [&_strong]:md:font-semibold font-sans text-sm leading-[1.125rem] md:text-base md:leading-[1.375rem] text-slate-400">
                {description}
              </h2>
            </div>
          </div>
        </div>
      </div>
    </li>
  );
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{ result: string; confidence: number; label?: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (selectedFile: File) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    } else {
      setError('Please upload a valid image file.');
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files?.[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000';
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        setError(errorData.error || 'Failed to analyze image. Ensure backend is running.');
        return;
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Something went wrong.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#030303] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200 overflow-x-hidden font-sans">

      {/* 1. Hero Section (Empty children) */}
      <HeroGeometric
        badge="AI-Powered Security"
        title1="Currency"
        title2="Guard"
      />

      {/* 2. Features Grid Demo */}
      <div className="container mx-auto px-6 py-20">
        <h2 className="text-4xl font-bold text-center mb-12 text-slate-200 tracking-tight">System Capabilities</h2>
        <ul className="grid grid-cols-1 gap-6 md:grid-cols-12 md:grid-rows-3 lg:gap-6 xl:max-h-[34rem] xl:grid-rows-2">
          <GridItem
            area="md:[grid-area:1/1/2/7] xl:[grid-area:1/1/2/5]"
            icon={<Box className="h-6 w-6 text-white" />}
            title="Deep Learning Core"
            description="Utilizes a robust ResNet50 architecture fine-tuned on thousands of currency samples."
          />
          <GridItem
            area="md:[grid-area:1/7/2/13] xl:[grid-area:2/1/3/5]"
            icon={<Zap className="h-6 w-6 text-white" />}
            title="Instant Analysis"
            description="Optimized inference pipeline delivers authentification results in milliseconds."
          />
          <GridItem
            area="md:[grid-area:2/1/3/7] xl:[grid-area:1/5/3/8]"
            icon={<Lock className="h-6 w-6 text-white" />}
            title="Secure Processing"
            description="Images are processed locally or securely in memory without permanent storage."
          />
          <GridItem
            area="md:[grid-area:2/7/3/13] xl:[grid-area:1/8/2/13]"
            icon={<Sparkles className="h-6 w-6 text-white" />}
            title="High Accuracy"
            description="Trained with focal loss to handle difficult edge cases and worn notes effectively."
          />
          <GridItem
            area="md:[grid-area:3/1/4/13] xl:[grid-area:2/8/3/13]"
            icon={<Search className="h-6 w-6 text-white" />}
            title="Visual Explanation"
            description="Confidence scores provide transparency into the model's decision making process."
          />
        </ul>
      </div>

      {/* 3. Upload Section */}
      <div className="w-full max-w-2xl mx-auto mb-40 text-left px-6">
        <h2 className="text-3xl font-bold text-center mb-8 text-slate-200 tracking-tight">Analyze Currency</h2>
        <div className="relative rounded-[1.5rem] border-[0.75px] border-slate-800 p-2 md:p-3">
          <GlowingEffect
            spread={40}
            glow={true}
            disabled={false}
            proximity={64}
            inactiveZone={0.01}
            borderWidth={3}
            variant="default"
          />

          <div className="relative bg-slate-950/80 backdrop-blur-xl rounded-xl border border-slate-800 p-8 md:p-12 shadow-2xl overflow-hidden">
            {/* Drop Zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "relative border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 min-h-[300px]",
                isDragOver ? "border-cyan-400 bg-cyan-500/5" : "border-slate-800 hover:border-slate-600 hover:bg-slate-900",
                preview ? "border-transparent p-0 overflow-hidden" : ""
              )}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                className="hidden"
                accept="image/*"
              />

              {preview ? (
                <div className="relative w-full h-full min-h-[300px] flex items-center justify-center bg-black/50 rounded-lg group">
                  <img src={preview} alt="Preview" className="max-w-full max-h-[300px] object-contain rounded-lg shadow-lg" />

                  {/* Hover Overlay with Change/Remove Options */}
                  <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center gap-3 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg backdrop-blur-sm">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        fileInputRef.current?.click();
                      }}
                      className="text-white font-medium flex items-center gap-2 bg-white/10 px-5 py-2.5 rounded-full hover:bg-white/20 border border-white/20 transition-all"
                    >
                      <ImageIcon className="w-5 h-5" /> Change Image
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setFile(null);
                        setPreview(null);
                        setResult(null);
                        setError(null);
                        if (fileInputRef.current) fileInputRef.current.value = '';
                      }}
                      className="text-red-400 font-medium flex items-center gap-2 bg-red-500/10 px-5 py-2.5 rounded-full hover:bg-red-500/20 border border-red-500/20 transition-all"
                    >
                      <XCircle className="w-5 h-5" /> Remove Image
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center space-y-6">
                  <div className="relative w-24 h-24 mx-auto">
                    <div className="absolute inset-0 bg-cyan-500/20 rounded-full blur-xl animate-pulse" />
                    <div className="relative w-full h-full bg-gradient-to-br from-slate-800 to-slate-900 rounded-full flex items-center justify-center border border-slate-700 group-hover:scale-105 transition-transform">
                      <Upload className="w-10 h-10 text-cyan-400" />
                    </div>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-slate-200">Upload Currency Note</p>
                    <p className="text-sm text-slate-500 mt-2">Drag & drop or click to browse</p>
                  </div>
                </div>
              )}
            </div>

            {/* Error Message */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-4 p-4 bg-red-950/30 border border-red-500/20 rounded-xl text-red-200 text-sm flex items-start gap-3"
                >
                  <XCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-red-400">Analysis Failed</p>
                    <p className="opacity-80">{error}</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Action Button */}
            {!result && (
              <button
                onClick={handleUpload}
                disabled={!file || isLoading}
                className={cn(
                  "w-full mt-8 py-4 rounded-xl font-bold text-lg tracking-wide transition-all duration-300 transform relative overflow-hidden group",
                  !file ? "bg-slate-900 text-slate-600 cursor-not-allowed border border-slate-800" :
                    isLoading ? "bg-slate-800 text-slate-400 cursor-wait border border-slate-700" :
                      "bg-white text-black hover:bg-slate-200 shadow-[0_0_20px_rgba(255,255,255,0.3)] hover:shadow-[0_0_30px_rgba(255,255,255,0.5)] border border-transparent"
                )}
              >
                {isLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <Loader2 className="w-5 h-5 animate-spin" /> Processing Analysis...
                  </span>
                ) : "Analyze for Authenticity"}
              </button>
            )}

            {/* Results Overlay/Section */}
            <AnimatePresence mode="wait">
              {result && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mt-8 space-y-6"
                >
                  <div className={cn(
                    "p-6 rounded-xl border backdrop-blur-md relative overflow-hidden flex items-center justify-between",
                    (result.label === 'Real' || result.result === 'Real')
                      ? "bg-emerald-500/10 border-emerald-500/50 shadow-[0_0_30px_rgba(16,185,129,0.2)]"
                      : "bg-red-500/10 border-red-500/50 shadow-[0_0_30px_rgba(239,68,68,0.2)]"
                  )}>
                    <div className="flex items-center gap-4">
                      {(result.label === 'Real' || result.result === 'Real') ? (
                        <div className="p-3 bg-emerald-500/20 rounded-full">
                          <CheckCircle2 className="w-8 h-8 text-emerald-400" />
                        </div>
                      ) : (
                        <div className="p-3 bg-red-500/20 rounded-full">
                          <XCircle className="w-8 h-8 text-red-400" />
                        </div>
                      )}
                      <div>
                        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest">Result</h3>
                        <p className={cn(
                          "text-3xl font-black tracking-tight",
                          (result.label === 'Real' || result.result === 'Real') ? "text-emerald-400" : "text-red-400"
                        )}>{result.result}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest">Confidence</h3>
                      <p className="text-3xl font-bold text-white">{(result.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>

                  <button
                    onClick={() => {
                      setFile(null);
                      setPreview(null);
                      setResult(null);
                      setError(null);
                      if (fileInputRef.current) fileInputRef.current.value = '';
                    }}
                    className="w-full py-4 rounded-xl font-bold text-lg tracking-wide transition-all duration-300 bg-slate-800 text-white hover:bg-slate-700 border border-slate-700"
                  >
                    Analyze New Image
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
