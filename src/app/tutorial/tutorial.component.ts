import { Component, OnInit, HostListener, ViewChild, ElementRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
import { BehaviorSubject } from 'rxjs';

import { MatSidenav } from '@angular/material';

import { Tutorial } from '../tutorial';
import { TOC } from '../toc';

@Component({
  selector: 'app-tutorial',
  templateUrl: './tutorial.component.html',
  styleUrls: ['./tutorial.component.css']
})
export class TutorialComponent implements OnInit {
  selectedTutorial: string;
  currentTutorialText: string;
  tutorialList: Tutorial[];
  tocContent: TOC[];

  minWidth: number = 640;
  screenWidth: number;
  private screenWidth$ = new BehaviorSubject<number>(window.innerWidth);

  @ViewChild('sidenav', { static: true })
  sidenav: MatSidenav;

  @ViewChild('grippy', { static: true })
  grippy: ElementRef;

  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.screenWidth$.next(event.target.innerWidth);
    if (this.sidenav.opened && this.screenWidth < this.minWidth) {
      this.grippy.nativeElement.style.backgroundImage = "url(../../assets/images/sidebar-grippy-show.png)"
      this.grippy.nativeElement.style.left = "0rem"
    }
    else{
      this.grippy.nativeElement.style.backgroundImage = "url(../../assets/images/sidebar-grippy-hide.png)"
      this.grippy.nativeElement.style.left = "20rem"
    }
  }

  constructor(private http: HttpClient, private router: Router, private route: ActivatedRoute) {
    this.route.params.subscribe(params => {
      if (params['name']) {
        this.selectedTutorial = params['name'];
        this.getTutorialStructure();
      }
    });

    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  ngOnInit() {
  }

  getTutorialStructure() {
    this.http.get('assets/tutorial/structure.json', {responseType: 'text'}).subscribe(data => {
      this.tutorialList = <Tutorial[]>JSON.parse(data);

      var t: Tutorial[] = this.tutorialList.filter(tutorial => tutorial.name === (this.selectedTutorial + ".md"));
      this.updateTutorialContent(t[0]);
    });
  }

  updateTutorialContent(tutorial: Tutorial) {
    if (!tutorial)
      this.router.navigate(['PageNotFound']);

    window.scroll(0,0);

    this.getSelectedTutorialText('assets/tutorial/' + tutorial.name);
  }

  getSelectedTutorialText(tutorialName) {
    this.http.get(tutorialName, {responseType: 'text'}).subscribe(data => {
      this.currentTutorialText = data;
    });
  }

  getImageUrl() {
    //console.log(this.sidenav.opened)
    if (this.sidenav.opened) {
      this.grippy.nativeElement.style.left = "20rem"
      return "url(../../assets/images/sidebar-grippy-hide.png)"
    }else{
      this.grippy.nativeElement.style.left = "0rem"
      return "url(../../assets/images/sidebar-grippy-show.png)"
    }
  }

  checkSidebar() {
    console.log(this.sidenav.opened)
    if (this.sidenav.opened) {
      this.grippy.nativeElement.style.backgroundImage = "url(../../assets/images/sidebar-grippy-hide.png)"
      this.grippy.nativeElement.style.left = "20rem"
    } else {
      this.grippy.nativeElement.style.backgroundImage = "url(../../assets/images/sidebar-grippy-show.png)"
      this.grippy.nativeElement.style.left = "0rem"
    }
  }

  toggleMenu(){
    this.sidenav.toggle();
    this.checkSidebar();
  }
}
