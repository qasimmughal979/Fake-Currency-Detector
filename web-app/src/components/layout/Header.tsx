import Link from 'next/link';
import { ShieldCheck, Github } from 'lucide-react';

export function Header() {
    return (
        <header className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-black/50 backdrop-blur-xl">
            <div className="container mx-auto px-6 h-16 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-2 group">
                    <div className="bg-cyan-500/10 p-2 rounded-lg border border-cyan-500/20 group-hover:border-cyan-500/40 transition-colors">
                        <ShieldCheck className="w-5 h-5 text-cyan-400 group-hover:text-cyan-300 transition-colors" />
                    </div>
                    <span className="font-semibold text-slate-200 tracking-tight group-hover:text-white transition-colors">
                        Currency<span className="text-cyan-400">Guard</span>
                    </span>
                </Link>

                <nav className="flex items-center gap-6">
                    <Link
                        href="https://github.com"
                        target="_blank"
                        className="text-sm text-slate-400 hover:text-white transition-colors flex items-center gap-2"
                    >
                        <Github className="w-4 h-4" />
                        <span className="hidden sm:inline">Source Code</span>
                    </Link>
                </nav>
            </div>
        </header>
    );
}
